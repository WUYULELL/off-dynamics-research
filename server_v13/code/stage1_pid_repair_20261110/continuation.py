"""Continue V13 stage 1 with verified GPU PID aliases and read-only reuse."""
import os
from pathlib import Path
import sys
import time

from admission import dmesg_tail, external_reason, reason
from common import read, sha, utc, write
import controller
import resources
from stage1_core import BASE, CONFIG, cache_receipt, publish


OLD = BASE/'outputs/stage1_20261110'
OLD_CODE = BASE/'stage1'
OUTPUT = BASE/'outputs/stage1_20261110_pid_repair'
PILOT = CONFIG['bc']['resource_pilot']
ORIGINAL_FREEZE = 'c40f78b87eb6276a1043ce30fa330529f2087424aa8f685f779014cb86af1c5d'
MAX_CONTEXT_START_SECONDS = 180
MAX_CONTEXT_MAPPING_SECONDS = 30


def completed_path(name):
    paths = controller.prior_attempts(OLD/'runs',name)
    assert paths, ('Original attempt does not exist',name)
    path = paths[-1]
    assert read(path/'OUTCOME.json')['status']=='complete', ('Original attempt is not complete',name)
    return path


def verify_original():
    # Called under the same stage lock as the original supervisor.
    failure = read(OLD/'FAILED.json')
    assert failure['type']=='AssertionError' and failure['stage2_started'] is False
    assert failure['message']=='All registered cards observed another process during our runs. See GPU_BLOCKED receipts.'
    assert not (OLD/'STAGE1_COMPLETED.json').exists() and not (OLD/'STAGE1_INCOMPLETE.json').exists()
    assert sha(OLD_CODE/'FROZEN.json')==ORIGINAL_FREEZE
    plan = read(OLD/'PLAN.json')
    assert plan['code_freeze_sha256']==ORIGINAL_FREEZE and plan['stage2_authorized'] is False
    assert sha(OLD/'config.json')==plan['config_sha256'] and read(OLD/'config.json')==CONFIG
    assert sha(OLD/'AUTHORIZATION.json')==plan['authorization_sha256']
    expected = {controller.identifier(v,s):[v,s] for s in CONFIG['bc']['seeds'] for v in CONFIG['bc']['variants']}
    assert plan['bc_jobs']==expected and plan['per_GPU_limit']==3
    assert [g['index'] for g in plan['registered_gpus']]==[3]
    assert all(not controller.prior_attempts(OLD/'runs',name) for name in expected if name!=PILOT)
    cache = completed_path(controller.CACHE)
    pilot = completed_path(PILOT)
    cache_receipt(cache,complete=True)
    audit = read(pilot/'AUDIT.json')
    completion = read(pilot/'COMPLETED.json')
    assert audit['passed'] and audit['CPU_only'] and audit['complete_sample_streams_verified']
    assert audit['completed_sha256']==sha(pilot/'COMPLETED.json')
    assert audit['checkpoint_sha256']==sha(pilot/'final_checkpoint.pt')==completion['checkpoint_sha256']
    assert sha(pilot/'steps.f32')==completion['steps_sha256']
    job = completion['job']
    assert job['job']==PILOT and job['seed']==601 and job['variant']=='BC-mix'
    assert job['gpu']['index']==3 and completion['steps']==CONFIG['bc']['steps']
    assert job['cache']==str(cache) and job['cache_receipt_sha256']==sha(cache/'CACHE_ACCEPTANCE.json')
    assert job['plan_sha256']==sha(OLD/'PLAN.json')
    launch = read(pilot/'LAUNCH.json')
    assert launch['job_sha256']==sha(pilot/'JOB.json')
    blocked = read(OLD/'GPU_BLOCKED_3.json')
    assert blocked['owned_pids']==[launch['pid']] and blocked['other_pids']
    assert blocked['sample']['gpu']['uuid']==job['gpu']['uuid']
    observed = [OLD/'FAILED.json',OLD/'GPU_BLOCKED_3.json',OLD/'PLAN.json',OLD/'config.json',OLD/'AUTHORIZATION.json',
                cache/'CACHE_ACCEPTANCE.json',cache/'CACHE_STATISTICS.json',cache/'OUTCOME.json',pilot/'AUDIT.json',
                pilot/'COMPLETED.json',pilot/'OUTCOME.json',pilot/'LAUNCH.json',pilot/'JOB.json']
    return dict(original_output=str(OLD),reused_cache=str(cache),reused_pilot=str(pilot),new_runs=24,
                original_block_ownership='unconfirmed across PID namespaces',
                evidence_sha256={str(path):sha(path) for path in observed},stage2_authorized=False)


def same_card(sample, expected):
    if 'error' in sample:
        return False
    return all(sample['gpu'][k]==expected[k] for k in ('index','uuid','driver'))


def discover_alias(sample, expected, prior):
    """A fresh, ready CUDA context must add exactly one compute PID."""
    assert same_card(sample,expected),'GPU identity became unverified during context attribution'
    assert prior<=set(sample['visible_pids']),'Previously attributed CUDA context disappeared'
    new = set(sample['visible_pids'])-prior
    assert len(new)<=1,'More than one new GPU PID during single-context attribution'
    if not new:
        return None
    pid = next(iter(new))
    matching = [x for x in sample['compute'] if x['pid']==pid and x['gpu_uuid']==expected['uuid']]
    if len(matching)!=1 or not matching[0]['memory_raw'].isdigit() or int(matching[0]['memory_raw'])<=0:
        return None
    return pid


class AliasQueue(controller.Queue):
    def __init__(self, *args):
        super().__init__(*args)
        self.origin = read(self.out/'CONTINUATION.json')
        self.aliases = {}

    def cpu_cache(self):
        path = Path(self.origin['reused_cache'])
        cache_receipt(path,complete=True)
        return path

    def reconcile(self, name):
        if name==PILOT:
            return Path(self.origin['reused_pilot'])
        return super().reconcile(name)

    def require_existing_contexts_idle(self, gpu):
        sample = super().snapshot(gpu)
        assert same_card(sample,gpu),'GPU identity became unverified before release'
        visible = set(sample['visible_pids'])
        assert visible==set(self.aliases), ('GPU contexts changed before release',sample)
        assert sample['gpu']['utilization_gpu_percent']==0, ('GPU is active before release',sample)
        if not visible:
            assert reason(sample,gpu,controller.POLICY) is None
        return sample

    def identify(self, item, gpu):
        job,folder,p = item
        ready = folder/'GPU_CONTEXT_READY.json'
        deadline = time.monotonic()+MAX_CONTEXT_START_SECONDS
        while not ready.is_file() and p.poll() is None and time.monotonic()<deadline:
            self.monitor_status('ATTRIBUTING_GPU_CONTEXT')
            time.sleep(.5)
        if p.poll() is not None:
            self.finish_before_mapping(item)
            return False
        assert ready.is_file(),'Timed out waiting for CUDA context readiness'
        local = read(ready)
        assert local['local_pid']==p.pid and local['gpu_uuid']==gpu['uuid'] and local['training_steps']==0
        assert local['CUDA_initialized'] is True
        deadline = time.monotonic()+MAX_CONTEXT_MAPPING_SECONDS
        candidate, earlier_sample = None, None
        while time.monotonic()<deadline:
            if p.poll() is not None:
                self.finish_before_mapping(item)
                return False
            observed = super().snapshot(gpu)
            host_pid = discover_alias(observed,gpu,set(self.aliases))
            if host_pid is not None and host_pid==candidate and observed['gpu']['utilization_gpu_percent']==0:
                assert host_pid not in self.aliases and p.pid not in self.aliases.values()
                self.aliases[host_pid] = p.pid
                write(folder/'GPU_PID_MAPPING.json',dict(time=utc(),driver_pid=host_pid,local_pid=p.pid,
                    new_compute_contexts=1,previous_owned_driver_pids=sorted(set(self.aliases)-{host_pid}),
                    prior_sample=earlier_sample,confirmed_sample=observed,local_readiness=local))
                self.event(state='GPU_context_attributed',job=job['job'],attempt=job['attempt'],gpu_index=gpu['index'],
                           driver_pid=host_pid,local_pid=p.pid)
                return True
            candidate,earlier_sample = host_pid,observed
            time.sleep(1)
        raise RuntimeError('Could not confirm exactly one new GPU driver PID for '+job['job'])

    def finish_before_mapping(self, item):
        job,folder,p = item
        p.wait()
        self.active.remove(item)
        failure = read(folder/'FAILED.json') if (folder/'FAILED.json').exists() else None
        gpu = job['gpu']
        current = super().snapshot(gpu)
        # Before a PID is attributed its own context is not evidence of external OOM.
        why = external_reason(p.returncode,failure,gpu,(),current)
        self.outcome(job['job'],folder,status='external_interruption' if why else 'failed',exitcode=p.returncode,
                     reason=why,gpu=gpu,GPU_training_steps=0,current_GPU=current,dmesg=dmesg_tail(),failure=failure)
        if not why:
            raise RuntimeError(f"{job['job']} exited before GPU PID attribution; see {folder}/stderr.log")

    def cancel_preparation(self, group, why):
        self.stop_own(group)
        for item in group:
            if item not in self.active:
                continue
            job,folder,p = item
            self.active.remove(item)
            assert not (folder/'FAILED.json').exists(), ('A waiting worker failed',folder)
            self.outcome(job['job'],folder,status='external_interruption',exitcode=p.returncode,
                         reason=why,gpu=job['gpu'],GPU_training_steps=0)

    def wave(self, selected, cache, projected_RSS):
        assert 1<=len(selected)<=3 and len({i for _,i in selected})==1
        assert not self.active
        self.aliases = {}
        gpu = self.cards[selected[0][1]]
        if not self.memory(projected_RSS)[0] or not self.admit(gpu):
            return False
        waves = self.out/'waves'
        waves.mkdir(exist_ok=True)
        folder = waves/f'wave_{len(list(waves.glob("wave_*")))+1:04d}'
        folder.mkdir()
        shared_release = folder/'RELEASE.json'
        group = []
        for name,_ in selected:
            if not self.memory(projected_RSS)[0]:
                self.cancel_preparation(group,'System memory floor before group CUDA release')
                return False
            prior = self.require_existing_contexts_idle(gpu)
            item = self.launch(name,gpu,cache)
            group.append(item)
            _,attempt_dir,p = item
            publish(attempt_dir/'PREPARE_CUDA.json',dict(time=utc(),gpu_uuid=gpu['uuid'],shared_release=str(shared_release),
                    prior_driver_pids=sorted(self.aliases),prior_GPU_sample=prior,local_pid=p.pid))
            if not self.identify(item,gpu):
                self.cancel_preparation(group,'Another owned worker was externally interrupted during CUDA preparation')
                return False
        if not self.memory(projected_RSS)[0]:
            self.cancel_preparation(group,'System memory floor before group CUDA release')
            return False
        sample = self.require_existing_contexts_idle(gpu)
        publish(shared_release,dict(time=utc(),gpu_uuid=gpu['uuid'],local_pids=sorted(self.aliases.values()),GPU_sample=sample))
        self.event(state='gpu_wave_released',jobs=[item[0]['job'] for item in group],gpu=gpu,
                   local_to_driver={str(p):h for h,p in self.aliases.items()})
        print('GPU',gpu['index'],'released',len(group),'concurrent BC jobs',flush=True)
        last_measurement = -30.
        last, external = None,set()
        own = {item[2].pid for item in group}
        while self.active:
            if time.monotonic()-last_measurement>=20.:
                last = self.snapshot(gpu)
                external.update(self.record_external(gpu['index'],last,set(self.aliases)))
                self.event(state='GPU_runtime_snapshot',gpu_index=gpu['index'],sample=last,
                           worker_RSS=[resources.owned_process(p) for p in sorted(own)])
                last_measurement = time.monotonic()
            for item in self.active[:]:
                if item[2].poll() is not None:
                    self.finish(item,external,last)
            self.monitor_status('BC_TRAINING')
            if self.active:
                time.sleep(2)
        return True


def main():
    if '--cluster-allows-six' in sys.argv:
        raise SystemExit('This continuation has a fixed maximum of 3 jobs on one registered GPU.')
    original_prepare = controller.prepare_output

    def prepare(out,args,freeze):
        assert out==OUTPUT and (args.resume or len(args.registered_gpus or ())==1)
        provenance = verify_original()
        if args.resume:
            assert provenance==read(out/'CONTINUATION.json'), 'Original evidence changed since first continuation'
        plan,session = original_prepare(out,args,freeze)
        assert len(plan['registered_gpus'])==1 and plan['per_GPU_limit']==3
        if not args.resume:
            write(out/'CONTINUATION.json',provenance)
        return plan,session

    controller.OUTPUT = OUTPUT
    controller.Queue = AliasQueue
    controller.prepare_output = prepare
    controller.main()


if __name__ == '__main__':
    main()
