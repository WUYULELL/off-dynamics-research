"""New-directory V14 CPU diagnostic controller. No SSH, GPU, training or retry."""
import argparse
import fcntl
import os
from pathlib import Path
import random
import sys
import time
import traceback

from core import HERE,read,sha,utc,write
from context import Inputs,inventory,verify_code
import limits


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers',type=int,default=16)
    args = parser.parse_args()
    assert 1<=args.workers<=24
    assert sys.platform=='linux' and sys.version_info[:3]==(3,8,13)
    assert Path(sys.prefix).resolve()==(Path.home()/'conda-envs/offdynamics38').resolve()
    assert os.environ['CUDA_VISIBLE_DEVICES']=='-1'
    verify_code()
    import importlib.metadata as metadata
    for name,version in read(HERE/'config.json')['expected_packages'].items():
        assert metadata.version(name)==version, ('Unexpected environment package',name)
    (HERE/'outputs').mkdir(exist_ok=True)
    lock = (HERE/'outputs/v14.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    tag = time.strftime('%Y%m%d_%H%M%S')
    out = HERE/'outputs'/('diagnostics_'+tag)
    out.mkdir()
    for name in ('A','B','C'):
        (out/name).mkdir()
    (HERE/'outputs/LATEST_RUN.txt').write_text(str(out)+'\n',encoding='utf-8')
    print('V14 output:',out,flush=True)
    started = time.monotonic()
    ticks = limits.cpu_ticks()
    before = None
    def heartbeat(phase,**kw):
        write(out/'STATUS.json',dict(time=utc(),phase=phase,**kw),replace=True)
        snapshot = limits.limits_snapshot()
        owned = limits.owned_process(os.getpid())
        with (out/'RESOURCE.jsonl').open('a',encoding='utf-8') as f:
            import json
            f.write(json.dumps(dict(time=utc(),phase=phase,parent=owned,
                effective_available_memory_bytes=snapshot['effective_available_memory_bytes']))+'\n')
        print(phase,kw,flush=True)
    try:
        heartbeat('VERIFYING_PROTECTED_INPUTS')
        before = inventory()
        write(out/'INPUT_HASHES_BEFORE.json',before)
        inputs = Inputs()
        random.seed(inputs.config['model_loading_seed'])
        __import__('numpy').random.seed(inputs.config['model_loading_seed'])
        inputs.freeze_plan(out)
        runtime = limits.limits_snapshot()
        write(out/'RESOURCES_BEFORE.json',runtime)
        # Imported binding is a new output copy; original conda files are not patched.
        _,_,private = inputs.legacy.prepare_private_binding(out)
        vendor = private.parent
        import mujoco_py
        assert private.resolve() in Path(mujoco_py.cymj.__file__).resolve().parents
        import diagnostic_b
        b0 = diagnostic_b.b0(inputs,out/'B',vendor,heartbeat)
        import diagnostic_c
        c = diagnostic_c.run(inputs,out/'C')
        heartbeat('C_COMPLETED')
        records = []
        if b0['passed']:
            cores = max(1,int(runtime['effective_visible_cpu_cores'])-2)
            records = diagnostic_b.source_tracks(inputs,out/'B',vendor,heartbeat,min(args.workers,cores))
        else:
            print('B0 failed its frozen scientific gate. B1/B2 not started. A and C continue.',flush=True)
        b = diagnostic_b.summarize(out/'B',b0,records)
        # After the B pool has fully exited, load CPU torch models; no fork after torch use.
        import diagnostic_a
        a = diagnostic_a.run(inputs,out/'A',heartbeat)
        heartbeat('VERIFYING_PROTECTED_INPUTS_AFTER')
        after = inventory()
        assert before==after, 'Protected original inputs changed'
        write(out/'INPUT_PROTECTION_AFTER.json',dict(passed=True,files=len(after),sha256_after=after))
        assert not __import__('torch').cuda.is_initialized()
        resource_result = dict(limits=runtime,elapsed_seconds=time.monotonic()-started,
            host_CPU_busy_percent=limits.busy_percent(ticks,limits.cpu_ticks()))
        write(out/'RESOURCES_AFTER.json',resource_result)
        import report
        report.build(out,inputs,a,b,c,resource_result)
        write(out/'RESULT.json',dict(A=a,B=b,C=c,resources=resource_result,new_training_updates=0,
            optimizer_instances=0,GPU_jobs=0,stage2_started=False))
        write(out/'COMPLETED.json',dict(completed=True,authorized_plan_stopped_with_reports=True,all_three_diagnostics_completed=bool(b0['passed']),
            A_completed=True,C_completed=True,B0_passed=b0['passed'],B1_B2_started=bool(records),
            status='AWAITING_USER_REVIEW' if b0['passed'] else 'B0_FAILED_A_C_COMPLETED_AWAITING_USER_REVIEW',
            own_child_workers_exited=True,new_training_updates=0,optimizer_instances=0,GPU_jobs=0,
            A_policy_evaluation_episodes=200,environment_steps=a['environment_steps']+b['simulator_steps'],
            CUDA_initialized=False,stage2_started=False,ended_utc=utc(),
            evidence_sha256={name:sha(out/name) for name in ('REPORT_ZH.md','RESULT.json','config.json','INPUT_HASHES_BEFORE.json','INPUT_PROTECTION_AFTER.json')}))
        heartbeat('STOPPED_AWAITING_USER_REVIEW')
        print('V14 DONE:',out/'REPORT_ZH.md',flush=True)
    except BaseException:
        failed = dict(time=utc(),traceback=traceback.format_exc(),automatic_retry=False,new_training_updates=0,GPU_jobs=0)
        if before:
            failed['protected_input_differences'] = [p for p,h in before.items() if not Path(p).is_file() or sha(p)!=h]
        write(out/'FAILED.json',failed)
        print(failed['traceback'],flush=True)
        raise


if __name__=='__main__':
    main()
