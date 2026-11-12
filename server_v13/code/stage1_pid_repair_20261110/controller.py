"""Authorized V13 Stage 1, from full CPU cache through a stopped BC delivery."""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
import traceback

from admission import dmesg_tail, external_reason, reason, snapshot
from common import (check_environment_protection, existing_environment_snapshot, full_smi, gpu_info,
                    private_binding, read, sha, utc, verify_package, verify_project, write)
import resources
from stage1_core import BASE, CONFIG, GIB, HERE, PROJECT, TASK, cache_receipt, publish, verify_runtime, verify_stage0


OUTPUT = BASE/'outputs/stage1_20261110'
POLICY = CONFIG['resource_policy']
CACHE = 'full_cache'


def sigterm(_signum, _frame):
    raise SystemExit(143)


def identifier(variant, seed):
    return f'{variant}_s{seed}'


def prior_attempts(root, name):
    folders = []
    for attempt in range(POLICY['external_retry_max']+1):
        p = root/(name if attempt == 0 else f'{name}_retry{attempt}')
        if p.exists():
            assert attempt == len(folders), 'Gap in attempt directories'
            folders.append(p)
    return folders


def compare_prefix(old, new):
    import numpy as np

    if not (old/'steps.f32').is_file():
        return dict(original=str(old),rerun=str(new),compared_steps=0,identical=None)
    data = (old/'steps.f32').read_bytes()
    width = 3*4
    n = len(data)//width
    assert n <= CONFIG['bc']['steps']
    if not n:
        return dict(original=str(old),rerun=str(new),compared_steps=0,identical=None,partial_tail_bytes=len(data))
    a = np.frombuffer(data[:n*width],dtype='<u4').reshape(n,3)
    with (new/'steps.f32').open('rb') as f:
        b = np.frombuffer(f.read(n*width),dtype='<u4').reshape(n,3)
    diff = np.any(a != b,axis=1)
    return dict(original=str(old),rerun=str(new),compared_steps=n,identical=not bool(diff.any()),
                first_differing_step=int(np.flatnonzero(diff)[0])+1 if diff.any() else None,
                steps_with_differences=int(diff.sum()),partial_tail_bytes=len(data)%width,
                original_log_sha256=hashlib.sha256(data).hexdigest(),new_log_sha256=sha(new/'steps.f32'))


class Queue:
    def __init__(self, out, plan, session, runtime):
        self.out = out
        self.plan = plan
        self.session = session
        self.runtime = runtime
        self.cards = {g['index']:g for g in plan['registered_gpus']}
        self.active = []
        self.blocked = {int(p.stem.split('_')[-1]) for p in out.glob('GPU_BLOCKED_*.json')}
        self.limits = resources.limits_snapshot()
        assert self.limits['cgroup_mapping_found'] and not self.limits['unreadable_fields']
        self.log = (out/'queue_events.jsonl').open('a',encoding='utf-8',buffering=1)
        self.host_start = resources.cpu_ticks()

    def event(self, **fields):
        self.log.write(json.dumps(dict(time=utc(),session=self.session['number'],**fields),ensure_ascii=False)+'\n')
        self.log.flush()
        os.fsync(self.log.fileno())

    def monitor_status(self, phase, pending=0):
        workers = []
        for job,folder,process in self.active:
            progress = folder/('progress.json' if job['job'] == CACHE else 'heartbeat.json')
            rec = read(progress) if progress.is_file() else None
            workers.append(dict(job=job['job'],pid=process.pid,physical_gpu=job['gpu']['index'] if job['gpu'] else None,
                                output=str(folder),heartbeat=rec))
        publish(self.out/'STATUS.json',dict(phase=phase,time=utc(),session=self.session['number'],pending=pending,
                                            blocked_gpu_indices=sorted(self.blocked),workers=workers))

    def memory(self, projected):
        mem = resources.key_values('/proc/meminfo')
        current = resources.limits_snapshot()['effective_available_memory_bytes']
        available = min(current,self.limits['effective_available_memory_bytes'])
        allow = mem['MemAvailable'] >= mem['MemTotal']*POLICY['host_memory_min_fraction']
        allow = allow and projected <= available*POLICY['memory_max_fraction']
        return allow,available

    def launch(self, name, gpu=None, cache=None):
        parent = self.out/'runs'
        previous = prior_attempts(parent,name)
        attempt = len(previous)
        assert attempt <= POLICY['external_retry_max']
        if previous:
            assert read(previous[-1]/'OUTCOME.json')['status'] == 'external_interruption'
        folder = parent/(name if attempt==0 else f'{name}_retry{attempt}')
        folder.mkdir()
        for d in ('tmp','cache','cuda'):
            (folder/d).mkdir()
        job = dict(stage=1,task=TASK,job=name,run=str(self.out),output=str(folder),attempt=attempt,
                   plan_sha256=sha(self.out/'PLAN.json'),controller_pid=os.getpid(),gpu=gpu,
                   session=self.session['number'],boot_id=self.session['boot_id'])
        if gpu is not None:
            variant, seed = self.plan['bc_jobs'][name]
            job.update(variant=variant,seed=seed,cache=str(cache),cache_receipt_sha256=sha(cache/'CACHE_ACCEPTANCE.json'),
                       private_binding=str(self.runtime/'private_runtime/mujoco_py'))
        config = folder/'JOB.json'
        write(config,job)
        if name==CACHE:
            command=[sys.executable,'-B','-u',str(HERE/'build_cache.py'),str(config)]
        else:
            command=[sys.executable,'-B','-u',str(HERE/'worker.py'),'train',str(config)]
        env = dict(os.environ,CUDA_VISIBLE_DEVICES=gpu['uuid'] if gpu else '-1',TMPDIR=str(folder/'tmp'),
                   XDG_CACHE_HOME=str(folder/'cache'),CYTHON_CACHE_DIR=str(folder/'cache/cython'),
                   CUDA_CACHE_PATH=str(folder/'cuda'))
        if gpu is not None:
            env['PYTHONPATH'] = str(self.runtime/'private_runtime')
        with (folder/'stdout.log').open('xb') as stdout, (folder/'stderr.log').open('xb') as stderr:
            process = subprocess.Popen(command,env=env,stdout=stdout,stderr=stderr,start_new_session=True)
        self.active.append((job,folder,process))
        write(folder/'LAUNCH.json',dict(pid=process.pid,time=utc(),job_sha256=sha(config),command=command,gpu=gpu,
                                       identity=resources.owned_process(process.pid)))
        self.event(state='started',job=name,attempt=attempt,pid=process.pid,gpu=gpu,output=str(folder))
        return job,folder,process

    def snapshot(self, gpu):
        try:
            return snapshot(gpu['index'])
        except (AssertionError,OSError,subprocess.SubprocessError) as exc:
            return dict(time=utc(),error=str(exc),gpu=gpu)

    def admit(self, gpu):
        sample = self.snapshot(gpu)
        refusal = sample.get('error') or reason(sample,gpu,POLICY)
        self.event(state='gpu_admission',gpu_index=gpu['index'],reason=refusal,sample=sample)
        return refusal is None

    def stop_own(self, items):
        for job,folder,p in items:
            if p.poll() is None:
                try:
                    assert os.getpgid(p.pid) == p.pid
                    os.killpg(p.pid,signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for _,_,p in items:
            try:
                p.wait(timeout=10)
            except subprocess.TimeoutExpired:
                if p.poll() is None:
                    try:
                        assert os.getpgid(p.pid) == p.pid
                        os.killpg(p.pid,signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                p.wait()

    def audited(self, name, folder):
        if name == CACHE:
            cache_receipt(folder,complete=True)
            return
        assert not (folder/'AUDIT_FAILED.json').exists()
        if not (folder/'AUDIT.json').exists():
            self.event(state='cpu_audit_started',job=name,output=str(folder))
            self.monitor_status('CPU_AUDIT')
            with (folder/'audit.stdout.log').open('ab') as stdout,(folder/'audit.stderr.log').open('ab') as stderr:
                env = dict(os.environ,CUDA_VISIBLE_DEVICES='-1')
                subprocess.run([sys.executable,'-B','-u',str(HERE/'worker.py'),'audit',str(folder/'JOB.json')],
                               env=env,stdout=stdout,stderr=stderr,check=True)
        receipt = read(folder/'AUDIT.json')
        assert receipt['passed'] and receipt['CPU_only'] and receipt['completed_sha256']==sha(folder/'COMPLETED.json')
        assert receipt['checkpoint_sha256']==sha(folder/'final_checkpoint.pt')
        earlier = prior_attempts(self.out/'runs',name)
        if len(earlier)>1 and not (folder/'PREFIX_COMPARISON.json').exists():
            reports = [compare_prefix(old,folder) for old in earlier if old!=folder]
            write(folder/'PREFIX_COMPARISON.json',dict(loss_and_diagnostics_only=True,comparisons=reports))

    def outcome(self, name, folder, **data):
        item = dict(time=utc(),job=name,**data)
        write(folder/'OUTCOME.json',item)
        self.event(state='outcome',output=str(folder),**{k:v for k,v in item.items() if k!='time'})

    def reconcile(self, name):
        folders = prior_attempts(self.out/'runs',name)
        if not folders:
            return None
        folder = folders[-1]
        if (folder/'OUTCOME.json').exists():
            state = read(folder/'OUTCOME.json')['status']
            if state=='complete':
                self.audited(name,folder)
                return folder
            assert state=='external_interruption', ('An earlier attempt failed',folder)
        else:
            assert self.session['number']>1, ('Unexpected unfinished attempt',folder)
            config = read(folder/'JOB.json') if (folder/'JOB.json').exists() else None
            same_boot_evidence = config is not None and config['session']<self.session['number'] and (
                self.out/f"SESSION_INTERRUPTED_{config['session']:04d}.json").is_file()
            assert config is None or config['boot_id']!=self.session['boot_id'] or same_boot_evidence, (
                'Live or ambiguous same-boot attempt')
            assert not (folder/'FAILED.json').exists() and not (folder/'AUDIT_FAILED.json').exists()
            complete = folder/('CACHE_ACCEPTANCE.json' if name==CACHE else 'COMPLETED.json')
            if complete.exists():
                self.audited(name,folder)
                self.outcome(name,folder,status='complete',exitcode=None,recovered_after_reboot=True)
                return folder
            why = 'controller received external SIGTERM' if same_boot_evidence else 'server boot ID changed'
            self.outcome(name,folder,status='external_interruption',exitcode=None,reason=why,
                         gpu=config['gpu'] if config else None)
        if len(folders)>POLICY['external_retry_max']:
            return 'exhausted'
        return None

    def finish(self, item, external=(), live_sample=None):
        job,folder,p = item
        name = job['job']
        p.wait()
        self.active.remove(item)
        code = p.returncode
        if code==0:
            self.audited(name,folder)
            self.outcome(name,folder,status='complete',exitcode=0,gpu=job['gpu'])
            return
        failure = read(folder/'FAILED.json') if (folder/'FAILED.json').exists() else None
        gpu = job['gpu']
        current = self.snapshot(gpu) if gpu else None
        unknown = set(external)
        if current and 'visible_pids' in current:
            aliases = getattr(self,'aliases',None)
            own = set(aliases) if aliases is not None else {t[2].pid for t in self.active}|{p.pid}
            unknown.update(set(current['visible_pids'])-own)
        why = external_reason(code,failure,gpu,unknown,current)
        if why == 'CUDA device became unavailable' and gpu['index'] not in self.blocked:
            self.blocked.add(gpu['index'])
            write(self.out/f"GPU_BLOCKED_{gpu['index']}.json",dict(time=utc(),physical_gpu=gpu['index'],reason=why,sample=current))
        status = 'external_interruption' if why else 'failed'
        self.outcome(name,folder,status=status,exitcode=code,reason=why,gpu=gpu,external_pids=sorted(unknown),
                     current_GPU=current,last_active_GPU=live_sample,dmesg=dmesg_tail(),failure=failure)
        if not why:
            raise RuntimeError(f'{name} failed with exit code {code}. See {folder}/stderr.log')

    def cpu_cache(self):
        while True:
            prior = self.reconcile(CACHE)
            if prior is not None:
                assert prior!='exhausted','Full-cache retry limit reached'
                return prior
            while not self.memory(GIB)[0]:
                self.event(state='resource_wait',phase='CACHE',seconds=30)
                self.monitor_status('WAITING_CPU_CACHE_MEMORY')
                time.sleep(30)
            item = self.launch(CACHE)
            while item[2].poll() is None:
                self.monitor_status('CPU_CACHE')
                time.sleep(2)
            self.finish(item)

    def record_external(self, index, sample, own):
        outside = set(sample.get('visible_pids',[]))-own
        if outside and index not in self.blocked:
            self.blocked.add(index)
            write(self.out/f'GPU_BLOCKED_{index}.json',dict(time=utc(),physical_gpu=index,owned_pids=sorted(own),
                                                          other_pids=sorted(outside),sample=sample))
            self.event(state='external_GPU_process_observed',gpu_index=index,other_pids=sorted(outside))
        return outside

    def wave(self, selected, cache, projected_RSS):
        group = {}
        for name,index in selected:
            if not self.memory(projected_RSS)[0]:
                break
            gpu = self.cards[index]
            if not self.admit(gpu):
                continue
            group.setdefault(index,[]).append(self.launch(name,gpu,cache))
        for index,items in list(group.items()):
            gpu = self.cards[index]
            fits, available = self.memory(projected_RSS)
            if not fits or not self.admit(gpu):
                # No RELEASE receipt was written, so these waiting processes never acquired CUDA.
                self.stop_own(items)
                for job,folder,p in items:
                    self.active.remove((job,folder,p))
                    self.outcome(job['job'],folder,status='external_interruption',exitcode=p.returncode,
                                 reason='GPU or system memory admission changed before CUDA release',gpu=gpu,
                                 memory_passed=fits,effective_available_memory_bytes=available,projected_RSS_bytes=projected_RSS)
                del group[index]
                continue
            for job,folder,p in items:
                publish(folder/'RELEASE.json',dict(time=utc(),gpu_uuid=gpu['uuid']))
                self.event(state='gpu_released',job=job['job'],attempt=job['attempt'],pid=p.pid,gpu=gpu)
        last, external = {}, {}
        last_measurement = -30.
        while self.active:
            if time.monotonic()-last_measurement>=20.:
                for index in group:
                    own = {p.pid for job,_,p in self.active if job['gpu']['index']==index}
                    if not own:
                        continue
                    sample = self.snapshot(self.cards[index])
                    last[index] = sample
                    own_wave = {p.pid for _,_,p in group[index]}
                    external.setdefault(index,set()).update(self.record_external(index,sample,own_wave))
                    self.event(state='GPU_runtime_snapshot',gpu_index=index,sample=sample,
                               worker_RSS=[resources.owned_process(pid) for pid in sorted(own)])
                last_measurement = time.monotonic()
            for item in self.active[:]:
                if item[2].poll() is not None:
                    index = item[0]['gpu']['index']
                    self.finish(item,external.get(index,()),last.get(index))
            self.monitor_status('BC_TRAINING')
            if self.active:
                time.sleep(2)
        return bool(group)

    def training(self, cache):
        results = {name:self.reconcile(name) for name in self.plan['bc_jobs']}
        pilot_order = [CONFIG['bc']['resource_pilot']]+[name for name in results
                       if name.startswith('BC-mix_s') and name!=CONFIG['bc']['resource_pilot']]
        pilot = next((results[name] for name in pilot_order if isinstance(results[name],Path)),None)
        while pilot is None:
            name = next((name for name in pilot_order if results[name] is None),None)
            assert name is not None,'All mixed BC resource pilots exhausted'
            index = self.preferred(name)
            if not self.wave([(name,index)],cache,4*GIB):
                self.wait('WAITING_BC_PILOT',len([x for x in results.values() if x is None]))
            results[name] = self.reconcile(name)
            if isinstance(results[name],Path):
                pilot = results[name]
        rss = read(pilot/'COMPLETED.json')['peak_RSS_bytes']
        estimate = int(rss*POLICY['gpu_rss_safety_factor'])
        parent = resources.owned_process(os.getpid())['RSS_bytes']
        slots = min(POLICY['all_gpu_max'],len(self.cards)*self.plan['per_GPU_limit'])
        if not (self.out/'BC_RSS_BEFORE_PARALLEL.json').exists():
            write(self.out/'BC_RSS_BEFORE_PARALLEL.json',dict(pilot=str(pilot),pilot_peak_RSS_bytes=rss,
                worker_estimate_bytes=estimate,controller_RSS_bytes=parent,maximum_slots=slots,
                projected_total_RSS_bytes=parent+slots*estimate,time=utc()))
        print('BC RSS:',rss,'bytes; estimated concurrency memory:',parent+slots*estimate,'bytes',flush=True)
        while any(value is None for value in results.values()):
            pending = [name for name,result in results.items() if result is None]
            counts = {i:0 for i in self.cards}
            selected = []
            for name in pending:
                index = self.preferred(name)
                if counts[index] >= self.plan['per_GPU_limit'] or len(selected)>=slots:
                    continue
                if not self.memory(parent+(len(selected)+1)*estimate)[0]:
                    break
                selected.append((name,index))
                counts[index] += 1
            if not selected or not self.wave(selected,cache,parent+len(selected)*estimate):
                self.wait('WAITING_BC_RESOURCES',len(pending))
            for name,_ in selected:
                results[name] = self.reconcile(name)
        return results

    def preferred(self, name):
        seed = self.plan['bc_jobs'][name][1]
        order = self.plan['seed_device_order'][str(seed)]
        selected = [i for i in order if i not in self.blocked]
        assert selected,'All registered cards observed another process during our runs. See GPU_BLOCKED receipts.'
        return selected[0]

    def wait(self, phase, pending):
        self.monitor_status(phase,pending)
        self.event(state='resource_wait',phase=phase,seconds=30,pending=pending)
        print(phase,'30 s, pending',pending,flush=True)
        time.sleep(30)

    def close(self):
        self.stop_own(self.active)
        self.log.close()


def prepare_output(out, args, freeze):
    boot_id = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if args.resume:
        assert out.is_dir() and not (out/'FAILED.json').exists()
        assert not (out/'STAGE1_COMPLETED.json').exists() and not (out/'STAGE1_INCOMPLETE.json').exists()
        plan = read(out/'PLAN.json')
        assert sha(out/'config.json') == plan['config_sha256']
        assert sha(out/'AUTHORIZATION.json') == plan['authorization_sha256']
        assert sha(HERE/'FROZEN.json') == plan['code_freeze_sha256']
        assert read(out/'config.json') == CONFIG and plan['stage2_authorized'] is False
        previous = sorted(out.glob('SESSION_[0-9][0-9][0-9][0-9].json'))
        assert previous, 'Stage 1 has no valid startup receipt'
        last = read(previous[-1])
        assert last['boot_id']!=boot_id or (out/f"SESSION_INTERRUPTED_{last['number']:04d}.json").is_file(), (
            'Resume requires a changed boot ID or a recorded external controller interruption')
        registered = [g['index'] for g in plan['registered_gpus']]
        assert not args.registered_gpus or args.registered_gpus == registered, 'Reuse the frozen registration list'
        assert not args.cluster_allows_six or plan['per_GPU_limit'] == 6
        assert sha(out/'PLAN.json') == last['plan_sha256']
        number = last['number']+1
    else:
        assert not out.exists(), 'Stage 1 output already exists. Preserve it and inspect STATUS.json.'
        assert args.registered_gpus and len(set(args.registered_gpus))==len(args.registered_gpus)
        assert len(args.registered_gpus)<=4
        cards = [{k:v for k,v in gpu_info(i).items() if k in ('index','uuid','name','driver','memory_total_MiB')}
                 for i in args.registered_gpus]
        assert len({g['uuid'] for g in cards}) == len(cards)
        assert shutil.disk_usage(out.parent).free >= 4*GIB, 'Keep at least 4 GiB available before starting'
        auth = read(HERE/'AUTHORIZATION.json')
        assert auth['stage']==1 and auth['authorized'] and not auth['stage2_authorized']
        n = len(cards)
        specs = {identifier(variant,seed):[variant,seed] for seed in CONFIG['bc']['seeds'] for variant in CONFIG['bc']['variants']}
        assert len(specs) == 25
        out.mkdir()
        (out/'runs').mkdir()
        shutil.copyfile(HERE/'config.json',out/'config.json')
        shutil.copyfile(HERE/'AUTHORIZATION.json',out/'AUTHORIZATION.json')
        plan = dict(stage=1,stage2_authorized=False,job_ids=[CACHE]+list(specs),bc_jobs=specs,registered_gpus=cards,
                    per_GPU_limit=POLICY['per_gpu_v13_max'] if args.cluster_allows_six else POLICY['per_gpu_default'],
                    global_GPU_limit=POLICY['all_gpu_max'],cluster_six_rule_user_attested=args.cluster_allows_six,
                    seed_device_order={str(seed):[cards[(i+k)%n]['index'] for k in range(n)]
                                       for i,seed in enumerate(CONFIG['bc']['seeds'])},
                    assignment_rule='cyclic registered order by seed; fall back in the fixed order if preferred device is blocked',
                    code_freeze_sha256=sha(HERE/'FROZEN.json'),config_sha256=sha(out/'config.json'),
                    authorization_sha256=sha(out/'AUTHORIZATION.json'),transfer_manifest_sha256=freeze['transfer_manifest_sha256'])
        write(out/'PLAN.json',plan)
        number = 1
    session = dict(number=number,pid=os.getpid(),boot_id=boot_id,time=utc(),plan_sha256=sha(out/'PLAN.json'))
    write(out/f'SESSION_{number:04d}.json',session)
    return plan,session


def run(args):
    freeze = verify_package()
    verify_stage0()
    runtime_gym = verify_runtime()
    assert os.environ.get('TMUX') and os.environ.get('CUDA_VISIBLE_DEVICES')=='-1'
    out = OUTPUT
    out.parent.mkdir(exist_ok=True)
    with (out.parent/'.stage1.lock').open('a') as lock,(out.parent/'.stage0.lock').open('rb') as old_lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        fcntl.flock(old_lock,fcntl.LOCK_SH|fcntl.LOCK_NB)
        plan,session = prepare_output(out,args,freeze)
        queue = None
        try:
            print('V13 stage 1 output:',out,flush=True)
            print('Verifying original inputs and environment',flush=True)
            before = verify_project(PROJECT,freeze['transfer_manifest_sha256'])
            environment = existing_environment_snapshot()
            write(out/f"ENVIRONMENT_{session['number']:04d}.json",environment)
            write(out/f"GPU_START_{session['number']:04d}.json",dict(smi=full_smi(),registered=plan['registered_gpus']))
            runtime = out/f"runtime_{session['number']:04d}"
            runtime.mkdir()
            private_binding(runtime)
            queue = Queue(out,plan,session,runtime)
            write(out/f"RESOURCES_{session['number']:04d}.json",queue.limits)
            cache = queue.cpu_cache()
            receipt = cache_receipt(cache,complete=True)
            assert receipt['filter']['threshold']==0.084
            print('Cache verified. Waiting for registered idle GPUs, then single BC-mix RSS pilot.',flush=True)
            accepted = queue.training(cache)
            queue.monitor_status('FINAL_CPU_REPORT')
            verify_stage0()
            verify_package()
            assert verify_project(PROJECT,freeze['transfer_manifest_sha256'])==before
            check_environment_protection(environment)
            assert all(sha(path)==digest for path,digest in runtime_gym.items())
            assert receipt == cache_receipt(cache,complete=True)
            write(out/'INPUT_PROTECTION_AFTER.json',dict(passed=True,stage0_preserved=True,transfer_files=len(before),
                 existing_environment_preserved=True,accepted_cache_receipt_sha256=sha(cache/'CACHE_ACCEPTANCE.json')))
            cpu = resources.busy_percent(queue.host_start,resources.cpu_ticks())
            write(out/f"CPU_USAGE_{session['number']:04d}.json",dict(nproc=queue.limits['nproc'],
                cgroup_effective_cpu=queue.limits['effective_visible_cpu_cores'],host_CPU_busy_percent=cpu))
            from stage1_report import deliver
            deliver(out,cache,accepted,plan)
            queue.monitor_status('AWAITING_STAGE2_CONFIRMATION' if all(isinstance(p,Path) for p in accepted.values())
                                 else 'STAGE1_INCOMPLETE')
        except SystemExit as exc:
            if exc.code != 143:
                raise
            if queue is not None:
                items = queue.active[:]
                queue.stop_own(items)
                for job,folder,process in items:
                    if (folder/'FAILED.json').exists() or (folder/'AUDIT_FAILED.json').exists():
                        queue.outcome(job['job'],folder,status='failed',exitcode=process.returncode,
                                      reason='worker failed before controller SIGTERM',gpu=job['gpu'])
                    elif (folder/('CACHE_ACCEPTANCE.json' if job['job']==CACHE else 'COMPLETED.json')).exists():
                        queue.audited(job['job'],folder)
                        queue.outcome(job['job'],folder,status='complete',exitcode=process.returncode,gpu=job['gpu'])
                    else:
                        queue.outcome(job['job'],folder,status='external_interruption',exitcode=process.returncode,
                                      reason='controller received external SIGTERM',gpu=job['gpu'])
                queue.active.clear()
            write(out/f"SESSION_INTERRUPTED_{session['number']:04d}.json",dict(time=utc(),reason='controller SIGTERM'))
            raise
        except (Exception,KeyboardInterrupt) as exc:
            write(out/'FAILED.json',dict(time=utc(),type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),stage2_started=False))
            raise
        finally:
            if queue is not None:
                queue.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scan',action='store_true')
    parser.add_argument('--registered-gpus',type=int,nargs='+',choices=range(4))
    parser.add_argument('--cluster-allows-six',action='store_true')
    parser.add_argument('--resume',action='store_true')
    args = parser.parse_args()
    if args.scan:
        for i in range(4):
            try:
                item = snapshot(i)
                print('GPU',i,item['gpu']['uuid'],reason(item,item['gpu'],POLICY) or 'idle; register before use',
                      'PIDs',item['visible_pids'])
            except (OSError,AssertionError,subprocess.SubprocessError) as exc:
                print('GPU',i,'unavailable or unidentifiable:',exc)
        return
    if not args.resume and not args.registered_gpus:
        parser.error('Register the intended GPU(s), then pass --registered-gpus INDEX [INDEX ...].')
    signal.signal(signal.SIGTERM,sigterm)
    run(args)


if __name__ == '__main__':
    main()
