"""Local serial V12 0.2 controller; hard stop before throughput or formal experiments."""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
sys.dont_write_bytecode=True
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import time
import traceback
from common import (HERE,read,write,sha,utc,verify_package,verify_project,verify_prior_receipt,
                    idle_gpu,full_smi,existing_environment_snapshot,check_environment_protection,
                    compare_loss_arrays,verify_failed_start,hash_tree)


def run_child(command,base,label,env):
    stdout=base/(label+'.stdout.log');stderr=base/(label+'.stderr.log')
    child=None
    with stdout.open('x',encoding='utf-8') as f, stderr.open('x',encoding='utf-8') as e:
        try:
            child=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=e,env=env,
                                   text=True,encoding='utf-8',bufsize=1)
            write(base/(label+'.launch.json'),dict(pid=child.pid,ppid=os.getpid(),command=command,
                CUDA_VISIBLE_DEVICES=env['CUDA_VISIBLE_DEVICES'],time=utc()))
            for line in child.stdout:
                f.write(line);f.flush();print(line,end='',flush=True)
            rc=child.wait()
            assert rc==0, '%s exited with %d; read %s'%(label,rc,stderr)
        finally:
            if child is not None and child.poll() is None:
                # Only the exact owned child object; never discover or signal other users' processes.
                child.terminate();child.wait()


def compare_runs(a,b):
    import numpy as np
    x=np.load(str(a/'losses.npy'),allow_pickle=False)
    y=np.load(str(b/'losses.npy'),allow_pickle=False)
    result=compare_loss_arrays(x,y)
    aa,bb=read(a/'AUDIT.json'),read(b/'AUDIT.json')
    initial_a,initial_b=read(a/'initialization.json'),read(b/'initialization.json')
    sa,sb=read(a/'sampling.json'),read(b/'sampling.json')
    result.update(final_parameter_bits_equal=aa['parameter_sha256']==bb['parameter_sha256'],
        full_checkpoint_state_equal=aa['full_checkpoint_state_sha256']==bb['full_checkpoint_state_sha256'],
        initialization_equal=initial_a==initial_b,target_stream_equal=sa==sb,
        serialized_loss_JSONL_equal=sha(a/'losses.jsonl')==sha(b/'losses.jsonl'),
        same_Linux_GPU_only=True,Windows_results_not_paired=True)
    result['passed']=all(result[k] for k in ('bitwise_equal','final_parameter_bits_equal',
        'full_checkpoint_state_equal','initialization_equal','target_stream_equal','serialized_loss_JSONL_equal'))
    if not result['final_parameter_bits_equal']:
        result['first_observed_parameter_mismatch_checkpoint']=10000
        result['earliest_parameter_mismatch_step']='Not identifiable from final-only parameter checkpoints'
    return result


def windows_inventory(root,out):
    # Do not substitute audit summaries or evaluations for missing raw per-step losses.
    manifest=read(root/'MANIFEST.json')['files']
    base='records/oracle_diag/results/coverage_raw_20261026_r14/runs/hopp_w0_s201/'
    present=[x['path'] for x in manifest if x['path'].startswith(base)]
    candidates=[p for p in present if p.endswith(('early_losses.json','steps.jsonl','losses.jsonl','losses.npy'))
                or '/tb/' in p]
    assert not candidates, 'Unexpected Windows raw losses found; stop to define the descriptive comparison'
    result=dict(status='无可对照记录',reason='迁移包未保留同配置 Hopper T seed201 的原始逐步损失或 TB 记录',
        matching_configuration=base+'config.json',available_records=present,
        audit_summaries_not_used_as_loss_input=True,relative_loss_difference_range=None,
        approximate_divergence_step=None,Windows_pairwise_inference=False)
    write(out/'WINDOWS_LOSS_COMPARISON.json',result)
    return result


def report(out,det,windows,entry,complete_jobs,error=None):
    lines=['# V12 阶段 0.2 GPU 学习器迁移验收报告','',
        '仅为迁移短跑，不是正式实验。Hopper T，seed201，温度10，target batch64，source不参与，cosine周期500000。',
        '三任务在同一张已分配的 A6000 上串行执行；两次10000步为同机确定性检查，另一次20000步包含两次各10回合评估。','',
        '## 已完成范围','',
        '|任务|学习器更新|Adam更新|训练含评估耗时 秒|评估回合|',
        '|---|---:|---:|---:|---:|']
    for name in complete_jobs:
        c=read(out/name/'COMPLETED.json')
        lines.append('|%s|%d|%d|%.3f|%d|'%(name,c['counts']['learner_updates'],c['counts']['Adam_updates'],
            c['elapsed_seconds'],c['evaluation_episodes']))
    lines+=['','## 同机确定性','',json.dumps(det,ensure_ascii=False,indent=2) if det else '尚未完成比较。',
        '', '比较原始逐步 V/Q/actor float32 数值位、JSONL 字节、最终四网络参数、三 Adam 和 cosine 完整状态。',
        '若只在最终参数发现差异，不将10000步冒充实际最早分岔步；以保存记录可识别的范围报告。',
        '', '## Windows 损失对照','', windows['status'] if windows else '未完成记录盘点。',
        '旧 config、audit 和评估结果不替代原始损失，也不与 Linux 新结果作同种子配对推断。']
    for name in complete_jobs:
        c=read(out/name/'COMPLETED.json')
        lines+=['','## '+name+' 损失值域','',
            '|损失|最小值|中位数|99分位|最大值|最后一步|','|---|---:|---:|---:|---:|---:|']
        for key,s in c['loss_statistics'].items():
            lines.append('|%s|%.9g|%.9g|%.9g|%.9g|%.9g|'%(key,s['minimum'],s['median'],s['p99'],s['maximum'],s['last']))
    if entry:
        lines+=['','## 完整入口与评估','',
            '|更新点|10回合平均分 描述性|长度均值|长度中位数|最短|长度小于200比例|',
            '|---|---:|---:|---:|---:|---:|']
        for ev in entry['events']:
            s=ev['length_statistics']
            lines.append('|%d|%.6f|%.3f|%.3f|%d|%.4f|'%(ev['step'],ev['mean'],s['mean'],s['median'],s['minimum'],s['fraction_under_200']))
        lines+=['','每回合原始回报、分数、长度、种子均保存。Python/NumPy/Torch CPU/CUDA 训练 RNG 与 sampler 状态在评估前后核对。',
            '没有五个末期评估点，Last5 不适用；这些短跑分数不用于正式效果或 H_root 判定。']
    lines+=['','## 当前停点','',
        '完成后 AWAITING_STAGE0_3_CONFIRMATION，不启动0.3、阶段1A/1B、ROOT环境或ROOT训练，也不启动任何旧队列。',
        '数值有限性、逐步确定性、完整状态、采样和评估记录接受自动核对；损失值域直接披露，不事后发明量级阈值。',
        '当前配置预先启用 deterministic_algorithms、关闭cudnn benchmark，TF32保持Torch1.11记录的默认开启状态；并非根据结果切换设置。',
        '旧输入与Conda保护见INPUT_PROTECTION_AFTER.json；正式阶段启动前仍需用户确认。']
    lines+=['','## 资源检查修复','',
        '本次是新目录中的资源准入修复。指定GPU允许最多3 MiB已观察到的常驻显存；连续3次利用率为0，且所选GPU的计算进程和pmon进程记录均须为空。',
        '其他GPU的进程不作为本卡占用。3 MiB的具体来源未被证明，只作为资源诊断中的常驻读数处理。',
        '原失败发生在创建训练任务之前；旧失败目录的完整哈希在前后核对。worker、science、audit与原版逐字节一致。']
    if error:lines+=['','## 故障或未通过','',error,'','已停止，不自动重试或调整确定性设置。']
    with (out/'REPORT_ZH.md').open('x',encoding='utf-8') as f:f.write('\n'.join(lines)+'\n')


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--project',type=Path,required=True)
    ap.add_argument('--prior',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--gpu',required=True)
    ap.add_argument('--allocation-confirmed',action='store_true')
    ap.add_argument('--confirm-resource-repair',action='store_true')
    args=ap.parse_args()
    assert args.allocation_confirmed,'Confirm the selected GPU is allocated to your account under cluster rules'
    assert args.confirm_resource_repair,'Manual confirmation of this concrete resource-gate repair is required; no automatic restart'
    assert sys.platform=='linux' and sys.version_info[:3]==(3,8,13)
    assert os.environ.get('TMUX'),'Run inside tmux'
    assert Path(sys.prefix).resolve()==(Path.home()/'conda-envs/offdynamics38').resolve()
    frozen=verify_package()
    root,out,prior=args.project.resolve(),args.output.resolve(),args.prior.resolve()
    assert out.parent==(Path.home()/'v12_linux/outputs').resolve() and out.name.startswith('stage0_2_')
    assert out.name=='stage0_2_20261107_resource_repair1'
    assert prior==(Path.home()/'v12_linux/outputs/stage0_1_20261106').resolve()
    out.mkdir(parents=True,exist_ok=False)
    env_before=None;inputs=None;det=None;windows=None;entry=None;done=[];old_start=None
    started=time.monotonic()
    try:
        cfg=read(HERE/'config_template.json')
        write(out/'config.json',dict(cfg,actual_project=str(root),actual_output=str(out),
            actual_prior=str(prior),selected_gpu=args.gpu,allocation_confirmed=True,started_utc=utc()))
        write(out/'CODE_FREEZE.json',frozen)
        old_dir=Path.home()/'v12_linux/outputs/stage0_2_20261106'
        old_start=verify_failed_start(old_dir)
        write(out/'ORIGINAL_FAILED_START_VERIFIED.json',old_start)
        write(out/'RESOURCE_REPAIR_AUTHORIZED.json',dict(explicit_manual_CLI_confirmation=True,
            confirmation_flag='--confirm-resource-repair',old_failure_sha256=old_start['old_failure_sha256'],
            scope='Resource admission repair only; original 0.2 three jobs unchanged',time=utc()))
        approval=verify_prior_receipt(prior);write(out/'STAGE0_1_APPROVAL.json',approval)
        gym_hashes=read(prior/'GYM_INPUT_HASHES.json')
        for path,h in gym_hashes.items():
            assert Path(sys.prefix).resolve() in Path(path).resolve().parents and sha(path)==h, 'Accepted Gym file changed'
        write(out/'GYM_INPUT_HASHES.json',gym_hashes)
        print('Checking original transfer and environment protection',flush=True)
        inputs=verify_project(root,frozen['project_manifest_sha256']);write(out/'INPUT_HASHES_BEFORE.json',inputs)
        env_before=existing_environment_snapshot();write(out/'ENVIRONMENT_BEFORE.json',env_before)
        for name,version in cfg['expected_versions'].items():
            import importlib.metadata as md
            assert md.version(name)==version,'Package version changed: '+name
        gpu=idle_gpu(args.gpu)
        write(out/'GPU_BEFORE.json',dict(selected=gpu,nvidia_smi=full_smi(),time=utc()))
        jobs={}
        for name,steps,evaluate in [('determinism_a',10000,False),('determinism_b',10000,False),('entry_20k',20000,True)]:
            job=dict(id=name,task='hopper-morph',seed=201,temp=10.,steps=steps,max_step=500000,
                evaluate=evaluate,project=str(root),gpu_uuid=gpu['uuid'],formal_training=False)
            path=out/(name+'.job.json');write(path,job);jobs[name]=sha(path)
        write(out/'WORKER_AUTHORIZATION.json',dict(scope='V12_0.2_three_fixed_jobs_only',jobs=jobs,
            total_learner_updates=40000,total_Adam_updates=120000,maximum_concurrent_GPU_workers=1,
            stage0_3_authorized=False,later_stages_authorized=False,automatic_resume=False))
        windows=windows_inventory(root,out)
        write(out/'PRE_RESULT_COMMITMENT.json',dict(config_sha256=sha(out/'config.json'),
            package_freeze_sha256=sha(HERE/'FROZEN.json'),jobs=jobs,gpu_uuid=gpu['uuid'],
            stage0_1_approved=True,before_any_GPU_learner_update=True,time=utc()))
        for name in ('determinism_a','determinism_b','entry_20k'):
            current=idle_gpu(gpu['uuid']);write(out/(name+'.GPU_ADMISSION.json'),dict(gpu=current,time=utc()))
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu['uuid']
            env['CUBLAS_WORKSPACE_CONFIG']=':4096:8';env['PYTHONUNBUFFERED']='1'
            print('Starting',name,'on',gpu['uuid'],flush=True)
            run_child([sys.executable,'-u','-B',str(HERE/'worker.py'),'--job',str(out/(name+'.job.json'))],out,name,env)
            audit_env=os.environ.copy();audit_env['CUDA_VISIBLE_DEVICES']='-1'
            run_child([sys.executable,'-u','-B',str(HERE/'audit.py'),'--run',str(out/name)],out,name+'.audit',audit_env)
            done.append(name)
            if name=='determinism_b':
                det=compare_runs(out/'determinism_a',out/'determinism_b')
                write(out/'DETERMINISM_COMPARISON.json',det)
                assert det['passed'],'Same-machine deterministic acceptance failed; no automatic retry or setting changes'
        import numpy as np
        a=np.load(str(out/'determinism_a/losses.npy'),allow_pickle=False)
        c=np.load(str(out/'entry_20k/losses.npy'),allow_pickle=False)
        prefix=compare_loss_arrays(a,c[:10000].copy())
        prefix['first_10000_parameter_and_stream_equal']=read(out/'determinism_a/FIRST_10000.json')==read(out/'entry_20k/FIRST_10000.json')
        write(out/'ENTRY_PREFIX_COMPARISON.json',prefix)
        assert prefix['bitwise_equal'] and prefix['first_10000_parameter_and_stream_equal'], 'Full entry changed the fixed pre-evaluation training prefix'
        entry=read(out/'entry_20k/evaluations.json')
        assert len(entry['events'])==2
        assert verify_project(root,frozen['project_manifest_sha256'])==inputs
        check_environment_protection(env_before)
        for path,h in gym_hashes.items():assert sha(path)==h,'Gym input changed during 0.2'
        assert sha(prior/'STAGE0_1_COMPLETED.json')==approval['receipt_sha256']
        assert hash_tree(old_dir)==old_start['old_files_sha256'],'Original failed start evidence changed'
        write(out/'INPUT_PROTECTION_AFTER.json',dict(passed=True,transfer_files=len(inputs),
            Conda_metadata_unchanged=True,installed_MuJoCo_binding_unchanged=True,
            MuJoCo_library_unchanged=True,stage0_1_receipt_unchanged=True,Gym_reference_files_unchanged=6))
        write(out/'ORIGINAL_FAILED_START_PROTECTION_AFTER.json',dict(passed=True,
            old_failure_sha256=old_start['old_failure_sha256'],files=len(old_start['old_files_sha256'])))
        report(out,det,windows,entry,done)
        completions=[read(out/n/'COMPLETED.json') for n in done]
        total_updates=sum(c['counts']['learner_updates'] for c in completions)
        assert total_updates==40000
        receipt=dict(completed=True,historical_technical_checks=True,
            acceptance_status='AWAITING_STAGE0_3_CONFIRMATION',jobs_completed=3,jobs_CPU_audited=3,
            learner_updates=total_updates,Adam_updates=120000,
            environment_steps=sum(c['counts']['constructor_environment_steps']+c['counts']['evaluation_environment_steps'] for c in completions),
            policy_evaluation_episodes=20,source_samples=0,maximum_concurrent_GPU_workers=1,
            physical_gpu_uuid=gpu['uuid'],deterministic_comparison_passed=det['passed'],
            windows_loss_comparison=windows['status'],full_entry_prefix_equal=True,
            elapsed_seconds=time.monotonic()-started,ended_utc=utc(),
            stage0_3_started=False,later_stages_started=False,automatic_retry_or_resume=False,
            formal_experiment_result=False)
        receipt['evidence_sha256']={p.relative_to(out).as_posix():sha(p) for p in out.rglob('*')
            if p.is_file() and 'private_runtime' not in p.relative_to(out).parts}
        write(out/'STAGE0_2_COMPLETED.json',receipt)
        print(json.dumps(receipt,ensure_ascii=False,indent=2),flush=True)
        print('Stopped. Read:',out/'REPORT_ZH.md',flush=True)
        return 0
    except BaseException:
        error=traceback.format_exc();protection={}
        try:
            if inputs is not None:protection['transfer_unchanged']=verify_project(root,frozen['project_manifest_sha256'])==inputs
            if env_before is not None:
                check_environment_protection(env_before);protection['environment_unchanged']=True
        except BaseException:protection['protection_check_error']=traceback.format_exc()
        if old_start is not None:
            protection['original_failed_start_unchanged']=hash_tree(old_dir)==old_start['old_files_sha256']
        write(out/'FAILED.json',dict(time=utc(),traceback=error,completed_audited_jobs=done,
            input_protection=protection,automatic_retry=False,stage0_3_started=False,later_stages_started=False))
        if not (out/'REPORT_ZH.md').exists():report(out,det,windows,entry,done,error)
        print(error,file=sys.stderr,flush=True)
        return 1


if __name__=='__main__':raise SystemExit(main())
