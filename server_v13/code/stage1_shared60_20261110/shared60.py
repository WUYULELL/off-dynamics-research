"""Shared-card continuation. The 60% rule applies only before CUDA admission."""
from pathlib import Path
import sys
import time

import admission
from common import read, sha, utc, write
from continuation import (AliasQueue, MAX_CONTEXT_MAPPING_SECONDS, MAX_CONTEXT_START_SECONDS,
                          PILOT, discover_alias, same_card, verify_original)
import controller
import resources
from stage1_core import BASE, CONFIG, HERE, publish


OUTPUT = BASE/'outputs/stage1_20261110_shared60'
EARLIER = BASE/'outputs/stage1_20261110_pid_repair'
RULE = read(HERE/'SHARED_POLICY.json')


def reuse_prior():
    proof = verify_original()
    reused = {PILOT:proof['reused_pilot']}
    if EARLIER.exists():
        assert not (EARLIER/'STAGE1_COMPLETED.json').exists(), 'Stage 1 already completed in the idle-card continuation'
        earlier = read(EARLIER/'FAILED.json')
        idle_block = 'All registered cards observed another process during our runs. See GPU_BLOCKED receipts.'
        assert earlier['type']=='KeyboardInterrupt' or (earlier['type']=='AssertionError' and earlier['message']==idle_block), (
            'Prior idle-card continuation needs review before this resource change')
        assert earlier['stage2_started'] is False
        plan = read(EARLIER/'PLAN.json')
        assert read(EARLIER/'config.json')==CONFIG and sha(EARLIER/'config.json')==plan['config_sha256']
        assert read(EARLIER/'CONTINUATION.json')==proof
        assert plan['stage2_authorized'] is False
        assert plan['bc_jobs']=={controller.identifier(v,s):[v,s] for s in CONFIG['bc']['seeds'] for v in CONFIG['bc']['variants']}
        for path in (EARLIER/'PLAN.json',EARLIER/'CONTINUATION.json',EARLIER/'FAILED.json',EARLIER/'config.json'):
            proof['evidence_sha256'][str(path)] = sha(path)
        for name in plan['bc_jobs']:
            attempts = controller.prior_attempts(EARLIER/'runs',name)
            if not attempts:
                continue
            assert name!=PILOT
            folder = attempts[-1]
            assert (folder/'OUTCOME.json').is_file() and read(folder/'OUTCOME.json')['status']=='complete', (
                'Prior attempt is incomplete. Preserve it and review its evidence before switching.',str(folder))
            audit,done,job = [read(folder/f) for f in ('AUDIT.json','COMPLETED.json','JOB.json')]
            assert audit['passed'] and audit['CPU_only'] and done['job']==job and job['job']==name
            assert [job['variant'],job['seed']]==plan['bc_jobs'][name] and done['steps']==CONFIG['bc']['steps']
            assert job['cache']==proof['reused_cache'] and job['plan_sha256']==sha(EARLIER/'PLAN.json')
            assert read(folder/'LAUNCH.json')['job_sha256']==sha(folder/'JOB.json')
            assert audit['completed_sha256']==sha(folder/'COMPLETED.json')
            assert audit['checkpoint_sha256']==done['checkpoint_sha256']==sha(folder/'final_checkpoint.pt')
            assert done['steps_sha256']==sha(folder/'steps.f32')
            for path in (folder/'AUDIT.json',folder/'COMPLETED.json',folder/'OUTCOME.json',folder/'JOB.json',folder/'LAUNCH.json'):
                proof['evidence_sha256'][str(path)] = sha(path)
            reused[name] = str(folder)
    assert len(reused)<25, 'All BC runs are already complete; use their existing delivery'
    proof.update(reused_BC=reused,new_runs=25-len(reused),shared_policy=RULE)
    return proof


def pre_admission_reason(sample, expected):
    if not same_card(sample,expected):
        return sample.get('error','GPU identity unavailable or changed')
    g = sample['gpu']
    if len(set(sample['visible_pids']))>RULE['max_existing_processes_per_gpu']:
        return 'more than 2 existing GPU processes'
    if not 0<=g['utilization_gpu_percent']<=RULE['admission_max_utilization_percent']:
        return 'utilization exceeds 60% before admission'
    if g['memory_total_MiB']-g['memory_used_MiB']<RULE['min_free_MiB']:
        return 'less than 12 GiB free GPU memory before admission'
    return None


def shared_external_reason(code, failure, gpu, others, current):
    # A previously admitted peer on its own is insufficient evidence of an external OOM.
    if failure is not None and gpu is not None and 'out of memory' in failure['message'].lower():
        known = current is not None and 'error' not in current
        low_memory = known and (current['gpu']['memory_total_MiB']-current['gpu']['memory_used_MiB']<RULE['min_free_MiB'])
        if not low_memory:
            others = ()
    return admission.external_reason(code,failure,gpu,others,current)


class SharedQueue(AliasQueue):
    def reconcile(self, name):
        if name in self.origin['reused_BC']:
            assert not controller.prior_attempts(self.out/'runs',name),'Reused BC run must not be duplicated'
            return Path(self.origin['reused_BC'][name])
        return super().reconcile(name)

    def admit_shared(self, gpu):
        samples = []
        for i in range(RULE['consecutive_admission_samples']):
            observed = self.snapshot(gpu)
            refusal = pre_admission_reason(observed,gpu)
            if refusal is None and samples and observed['visible_pids']!=samples[-1]['visible_pids']:
                refusal = 'existing GPU PID list changed between admission samples'
            samples.append(observed)
            if refusal is not None:
                self.event(state='shared_GPU_wait',gpu_index=gpu['index'],reason=refusal,samples=samples)
                return None
            if i+1<RULE['consecutive_admission_samples']:
                time.sleep(RULE['admission_sample_spacing_seconds'])
        self.event(state='shared_GPU_admitted',gpu_index=gpu['index'],samples=samples)
        return samples[-1]

    def block_ambiguous(self, gpu, sample, baseline, explanation):
        index = gpu['index']
        if index not in self.blocked:
            self.blocked.add(index)
            write(self.out/f'GPU_BLOCKED_{index}.json',dict(time=utc(),physical_gpu=index,reason=explanation,
                  admitted_peer_driver_pids=sorted(baseline),sample=sample))
            self.event(state='GPU_attribution_ambiguous',gpu_index=index,reason=explanation,sample=sample)

    def identify_shared(self, item, gpu, baseline):
        job,folder,p = item
        ready = folder/'GPU_CONTEXT_READY.json'
        deadline = time.monotonic()+MAX_CONTEXT_START_SECONDS
        while not ready.is_file() and p.poll() is None and time.monotonic()<deadline:
            self.monitor_status('ATTRIBUTING_GPU_CONTEXT')
            time.sleep(.5)
        if p.poll() is not None:
            self.finish_before_mapping(item)
            return None
        assert ready.is_file(),'Timed out waiting for CUDA context readiness'
        local = read(ready)
        assert local['local_pid']==p.pid and local['gpu_uuid']==gpu['uuid']
        assert local['training_steps']==0 and local['CUDA_initialized'] is True
        deadline = time.monotonic()+MAX_CONTEXT_MAPPING_SECONDS
        candidate, earlier = None,None
        while time.monotonic()<deadline:
            if p.poll() is not None:
                self.finish_before_mapping(item)
                return None
            observed = self.snapshot(gpu)
            if not same_card(observed,gpu):
                self.block_ambiguous(gpu,observed,baseline,'GPU disappeared before CUDA context attribution')
                return None
            visible = set(observed['visible_pids'])
            if len(visible-baseline)>1:
                self.block_ambiguous(gpu,observed,baseline,'More than one new PID while staging one worker')
                return None
            if not baseline<=visible:
                self.event(state='shared_GPU_peer_set_changed_during_mapping',gpu_index=gpu['index'],sample=observed)
                return None
            host = discover_alias(observed,gpu,baseline)
            if host is not None and host==candidate:
                assert host not in self.aliases and p.pid not in self.aliases.values()
                self.aliases[host] = p.pid
                write(folder/'GPU_PID_MAPPING.json',dict(time=utc(),driver_pid=host,local_pid=p.pid,
                      admitted_peer_driver_pids=sorted(baseline),prior_sample=earlier,confirmed_sample=observed,local_readiness=local))
                self.event(state='GPU_context_attributed',job=job['job'],attempt=job['attempt'],gpu_index=gpu['index'],
                           driver_pid=host,local_pid=p.pid)
                return host
            candidate,earlier = host,observed
            time.sleep(1)
        raise RuntimeError('Could not confirm a new CUDA driver PID for '+job['job'])

    def release_allowed(self, gpu, sample, own, baseline):
        if not same_card(sample,gpu):
            self.block_ambiguous(gpu,sample,baseline,'GPU disappeared before learner release')
            return False
        allowed = {own}|baseline
        foreign = self.record_external(gpu['index'],sample,allowed)
        return not foreign and own in sample['visible_pids']

    def wave(self, selected, cache, projected_RSS):
        assert 1<=len(selected)<=3 and len({i for _,i in selected})==len(selected)
        assert not self.active
        self.aliases = {}
        waves = self.out/'waves'
        waves.mkdir(exist_ok=True)
        group = {}
        wave_dir = None
        for name,index in selected:
            assert index in RULE['permitted_gpu_indices']
            if not self.memory(projected_RSS)[0]:
                break
            gpu = self.cards[index]
            admitted = self.admit_shared(gpu)
            if admitted is None or not self.memory(projected_RSS)[0]:
                continue
            if wave_dir is None:
                wave_dir = waves/f'wave_{len(list(waves.glob("wave_*")))+1:04d}'
                wave_dir.mkdir()
            folder = wave_dir/f'gpu_{index}'
            folder.mkdir()
            release = folder/'RELEASE.json'
            baseline = set(admitted['visible_pids'])
            item = self.launch(name,gpu,cache)
            _,attempt,p = item
            publish(attempt/'PREPARE_CUDA.json',dict(time=utc(),gpu_uuid=gpu['uuid'],shared_release=str(release),
                    prior_driver_pids=sorted(baseline),prior_GPU_sample=admitted,local_pid=p.pid))
            own = self.identify_shared(item,gpu,baseline)
            if own is None:
                self.cancel_preparation([item],'GPU peer or device changed before ownership was established')
            else:
                group[index] = item,own,baseline,release
        for index,(item,own,baseline,release) in list(group.items()):
            gpu = self.cards[index]
            sample = self.snapshot(gpu)
            fits,_ = self.memory(projected_RSS)
            if not fits or not self.release_allowed(gpu,sample,own,baseline):
                self.cancel_preparation([item],'Device, process set, or system memory changed before learner release')
                del group[index]
                continue
            # Utilization is intentionally not evaluated after initial CUDA admission.
            publish(release,dict(time=utc(),gpu_uuid=gpu['uuid'],local_pids=[item[2].pid],GPU_sample=sample))
            self.event(state='shared_BC_released',job=item[0]['job'],gpu_index=index,driver_pid=own,
                       local_pid=item[2].pid,admitted_peer_driver_pids=sorted(baseline))
        if group:
            print('Shared GPU wave released:',[(i,item[0]['job']) for i,(item,*_) in group.items()],flush=True)
        last_measurement = -30.
        last,external = {},{}
        while self.active:
            if time.monotonic()-last_measurement>=20.:
                for index,(item,own,baseline,_) in group.items():
                    if item not in self.active:
                        continue
                    sample = self.snapshot(self.cards[index])
                    last[index] = sample
                    external.setdefault(index,set()).update(self.record_external(index,sample,{own}|baseline))
                    self.event(state='GPU_runtime_snapshot',gpu_index=index,sample=sample,
                               worker_RSS=[resources.owned_process(item[2].pid)])
                last_measurement = time.monotonic()
            for item in self.active[:]:
                if item[2].poll() is not None:
                    index = item[0]['gpu']['index']
                    self.finish(item,external.get(index,()),last.get(index))
            self.monitor_status('BC_TRAINING')
            if self.active:
                time.sleep(2)
        return bool(group)


def main():
    if '--cluster-allows-six' in sys.argv:
        raise SystemExit('This continuation permits 1 added process per registered GPU, up to 3 in total.')
    if sys.argv[1:]==['--scan']:
        for index in RULE['permitted_gpu_indices']:
            try:
                sample = admission.snapshot(index)
                detail = pre_admission_reason(sample,sample['gpu']) or 'single sample passes; registration and repeated checks required'
                print('GPU',index,detail,'driver PIDs',sample['visible_pids'],flush=True)
            except (AssertionError,OSError,controller.subprocess.SubprocessError) as exc:
                print('GPU',index,'unverified:',exc,flush=True)
        return
    original_prepare = controller.prepare_output

    def prepare(out,args,freeze):
        assert out==OUTPUT
        assert set(args.registered_gpus or ()).issubset(RULE['permitted_gpu_indices'])
        if args.registered_gpus:
            args.registered_gpus = sorted(args.registered_gpus,key=lambda i:(i!=3,i))
        proof = reuse_prior()
        if args.resume:
            assert proof==read(out/'CONTINUATION.json'), 'Original evidence changed since first shared continuation'
        plan,session = original_prepare(out,args,freeze)
        assert set(g['index'] for g in plan['registered_gpus']).issubset(RULE['permitted_gpu_indices'])
        assert plan['per_GPU_limit']==1 and plan['global_GPU_limit']==3
        assert plan['GPU_admission']==RULE
        if not args.resume:
            write(out/'CONTINUATION.json',proof)
        print('Shared admission mode: 60% before startup; already audited BC jobs reused:',len(proof['reused_BC']),
              '; remaining:',proof['new_runs'],flush=True)
        return plan,session

    controller.POLICY = dict(controller.POLICY,per_gpu_default=1,all_gpu_max=3)
    controller.external_reason = shared_external_reason
    controller.OUTPUT = OUTPUT
    controller.Queue = SharedQueue
    controller.prepare_output = prepare
    controller.main()


if __name__ == '__main__':
    main()
