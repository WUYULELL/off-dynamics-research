"""V12 0.3 bounded single-GPU throughput measurement, with a terminal stop."""
import os
import sys
os.environ['CUDA_VISIBLE_DEVICES']='-1';os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
sys.dont_write_bytecode=True
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import time
import traceback
from common import (HERE,read,write,publish,sha,utc,verify_package,verify_project,verify_stage02,hash_tree,
                    idle_gpu,gpu_info,full_smi,existing_environment_snapshot,check_environment_protection)
from resources import limits_snapshot,tier_allowed,Monitor,group_rates


def run_tier(base,name,job_paths,limits,config,gpu):
    folder=base/'tiers'/name;folder.mkdir()
    processes=[];streams=[];phase='initialization';start=time.monotonic();release=None;eval_released=False
    monitor=Monitor(limits,config['resource_limits'],lambda:gpu_info(gpu['uuid']))
    try:
        for path in job_paths:
            job=read(path);stdout=(base/'logs'/(job['id']+'.stdout.log')).open('x',encoding='utf-8')
            stderr=(base/'logs'/(job['id']+'.stderr.log')).open('x',encoding='utf-8');streams.extend([stdout,stderr])
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu['uuid'];env['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
            p=subprocess.Popen([sys.executable,'-u','-B',str(HERE/'worker.py'),'--job',str(path)],stdout=stdout,stderr=stderr,env=env)
            processes.append(p)
            write(folder/(job['id']+'.launch.json'),dict(pid=p.pid,ppid=os.getpid(),job_sha256=sha(path),
                GPU_UUID=gpu['uuid'],time=utc()))
        runs=[base/'runs'/read(p)['id'] for p in job_paths]
        last_print=0.;last_monitor=0.
        with (folder/'resource_samples.jsonl').open('x',encoding='utf-8') as log:
            while True:
                now=time.monotonic()
                for p,run in zip(processes,runs):
                    rc=p.poll()
                    if rc is not None:
                        assert rc==0,'%s worker exited %d; preserve its FAILED/progress/stderr'%(run.name,rc)
                        assert (run/'COMPLETED.json').is_file(),'Worker exited without a completion receipt'
                if release is None and all((p/'READY.json').exists() for p in runs):
                    for p,r in zip(processes,runs):assert read(r/'READY.json')['pid']==p.pid
                    release=time.monotonic();publish(folder/'TRAIN_START.json',dict(monotonic=release,time=utc()))
                    phase='training';print(name,'all workers ready; training started',flush=True)
                if release is not None and not eval_released and all((p/'TRAINING_DONE.json').exists() for p in runs):
                    publish(folder/'EVALUATION_START.json',dict(monotonic=time.monotonic(),time=utc()))
                    eval_released=True;phase='evaluation';print(name,'all training complete; evaluation started',flush=True)
                if now-last_monitor>=config['monitor_interval_seconds']:
                    sample=monitor.sample(processes,phase);log.write(json.dumps(sample,ensure_ascii=False,allow_nan=False)+'\n');log.flush();last_monitor=now
                if now-last_print>=15:
                    progress=[]
                    for r in runs:
                        files=sorted(r.glob('progress_*.json')) if r.exists() else []
                        progress.append(dict(job=r.name,step=read(files[-1])['step'] if files else 0))
                    print(name,phase,progress,flush=True);last_print=now
                if all(p.poll()==0 for p in processes):break
                assert now-start<config['tier_timeout_seconds'],'Tier exceeded maximum allowed time'
                if release is None:assert now-start<config['initialization_timeout_seconds'],'Workers did not reach initialization barrier'
                time.sleep(.2)
        assert release is not None and eval_released
        done=[read(p/'COMPLETED.json') for p in runs]
        rates=group_rates(done,release)
        gpu_samples=[x['gpu'] for x in monitor.observed if x['phase']=='training']
        cpu_samples=[x for x in monitor.observed if x['phase']=='training']
        def mean(values):return sum(values)/len(values) if values else None
        metrics=dict(name=name,algorithm=done[0]['job']['algorithm'],concurrency=len(runs),
            worker_results=[dict(job=x['job']['id'],seed=x['job']['seed'],first_10000_seconds=x['first_10000_seconds'],
                second_10000_seconds=x['second_10000_seconds'],training_seconds=x['training_seconds'],
                evaluation_seconds=x['evaluation_seconds'],length_statistics=x['length_statistics']) for x in done],
            rates=rates,mean_GPU_utilization_percent=mean([x['utilization_gpu_percent'] for x in gpu_samples]),
            max_GPU_memory_used_MiB=max(x['gpu']['memory_used_MiB'] for x in monitor.observed),
            mean_owned_CPU_percent_one_core_100=mean([x['owned_CPU_percent_one_core_100'] for x in cpu_samples]),
            mean_host_CPU_busy_percent=mean([x['host_CPU_busy_percent'] for x in cpu_samples if x['host_CPU_busy_percent'] is not None]),
            max_worker_RSS_bytes=monitor.max_worker_RSS,pressure_warnings=monitor.warnings,
            training_sample_count=len(gpu_samples),monitor_interval_seconds=config['monitor_interval_seconds'],
            tier_total_wall_seconds=time.monotonic()-start,metrics_include_monitor_and_logging_overhead=True)
        write(folder/'MEASURED.json',metrics)
        # GPU workers have exited; each audit is a separate CPU-only child.
        for run in runs:
            env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='-1'
            with (base/'logs'/(run.name+'.audit.stdout.log')).open('x',encoding='utf-8') as out, (base/'logs'/(run.name+'.audit.stderr.log')).open('x',encoding='utf-8') as err:
                subprocess.run([sys.executable,'-B',str(HERE/'audit.py'),'--run',str(run)],env=env,stdout=out,stderr=err,check=True)
            assert read(run/'AUDIT.json')['passed']
        write(folder/'TIER_COMPLETED.json',dict(passed=True,workers=len(runs),metrics_sha256=sha(folder/'MEASURED.json'),
            all_CPU_audits_passed=True,time=utc()))
        return metrics,done
    finally:
        for p in processes:
            if p.poll() is None:p.terminate()
        for p in processes:
            if p.poll() is None:p.wait()
        for s in streams:s.close()


def analyze(measured):
    iql=[x for x in measured if x['algorithm']=='IQL']
    safe=[x for x in iql if not x['pressure_warnings']]
    best=max(safe,key=lambda x:x['rates']['aggregate_group_updates_per_second']) if safe else None
    estimates=[]
    for x in measured:
        multiplier=25 if x['algorithm']=='IQL' else 10
        points=50 if x['algorithm']=='IQL' else 40
        train=sum(w['training_seconds'] for w in x['worker_results'])/x['concurrency']
        evaluation=sum(w['evaluation_seconds'] for w in x['worker_results'])/x['concurrency']
        estimates.append(dict(algorithm=x['algorithm'],concurrency=x['concurrency'],
            full_training_updates=500000 if x['algorithm']=='IQL' else 200000,
            training_only_seconds=multiplier*train,estimated_total_seconds=multiplier*train+points*evaluation,
            evaluation_points=points,uses_early_policy_episode_length=True,not_a_runtime_guarantee=True))
    return dict(recommended_measured_concurrency=None if best is None else best['concurrency'],
        selection='Highest group updates/s among measured IQL tiers without resource-pressure flags',
        future_concurrency_requires_user_confirmation=True,not_a_formal_experiment_authorization=True,
        estimates=estimates,limitation='One short measurement per tier under shared background load; early short episodes can underestimate late evaluation time')


def report(base,measured,skipped,analysis,error=None):
    rows=['# V12 阶段 0.3 单卡吞吐测量报告','',
        'FAILED 已停止，不自动重试。' if error else '测速已停止，等待用户决定后续实验。',
        '所有结果均为丢弃的测速短跑，不计入正式实验，不按回报选择参数或并发数。','',
        '|算法|并发|整组更新每秒|各进程速率之和|平均GPU利用率|显存峰值 MiB|本任务CPU百分比 100代表一核|',
        '|---|---:|---:|---:|---:|---:|---:|']
    for x in measured:
        f=lambda v:'未记录' if v is None else '%.3f'%v
        rows.append('|%s|%d|%s|%s|%s|%d|%s|'%(x['algorithm'],x['concurrency'],f(x['rates']['aggregate_group_updates_per_second']),
            f(x['rates']['sum_individual_updates_per_second']),f(x['mean_GPU_utilization_percent']),x['max_GPU_memory_used_MiB'],f(x['mean_owned_CPU_percent_one_core_100'])))
    for x in measured:
        rows+=['','## '+x['name'],'',
            '|种子|前1万步 秒|后1万步 秒|训练总计 秒|一次10回合评估 秒|长度均值|长度中位数|最短|小于200比例|',
            '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
        for w in x['worker_results']:
            l=w['length_statistics']
            rows.append('|%d|%.3f|%.3f|%.3f|%.3f|%.3f|%.3f|%d|%.4f|'%(w['seed'],w['first_10000_seconds'],w['second_10000_seconds'],
                w['training_seconds'],w['evaluation_seconds'],l['mean'],l['median'],l['minimum'],l['fraction_under_200']))
        rows+=['','资源压力标记：'+json.dumps(x['pressure_warnings'],ensure_ascii=False)]
    rows+=['','## 未测档位','',json.dumps(skipped,ensure_ascii=False,indent=2),
        '', '## 建议和完整训练耗时外推','',json.dumps(analysis,ensure_ascii=False,indent=2),
        '', '推荐仅针对本次已测且资源压力可接受的IQL档位；未来正式并发仍待用户确认。',
        '整组吞吐按总更新数/同步放行到最后训练结束的时间计算；另一列是各进程自身更新速率之和，两者不混为一谈。',
        '训练计时不包含初始化、环境创建、屏障等待、评估和CPU审计；每进程训练耗时包含相同的逐步损失记录开销。',
        'IQL500k估计采用25倍20k训练时间加50次评估；BC200k采用10倍训练时间加40次评估。早期短回合可能低估后期评估耗时。',
        '本次只有末尾一次评估，Last5不适用。没有重新训练正式种子或启动阶段1A/1B/ROOT。',
        '', '## 停止位置','',
        'FAILED' if error else ('AWAITING_USER_DECISION_RESOURCE_LIMITED' if skipped else 'AWAITING_USER_STAGE1_DECISION'),
        '原迁移项目、原Conda、已完成0.2与原代码保持只读，详见保护凭据。']
    if error:rows+=['','## 故障','',error]
    with (base/'REPORT_ZH.md').open('x',encoding='utf-8') as f:f.write('\n'.join(rows)+'\n')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--project',type=Path,required=True);ap.add_argument('--prior',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True);ap.add_argument('--gpu',required=True);ap.add_argument('--allocation-confirmed',action='store_true')
    args=ap.parse_args();assert args.allocation_confirmed and os.environ.get('TMUX')
    assert sys.platform=='linux' and sys.version_info[:3]==(3,8,13)
    assert Path(sys.prefix).resolve()==(Path.home()/'conda-envs/offdynamics38').resolve()
    freeze=verify_package();cfg=read(HERE/'config_template.json')
    root,prior,base=args.project.resolve(),args.prior.resolve(),args.output.resolve()
    assert base==(Path.home()/'v12_linux/outputs/stage0_3_20261107').resolve()
    base.mkdir(parents=True,exist_ok=False)
    measured=[];skipped=[];completions=[];old_inputs=None;old_env=None;prior_hashes=None;started=time.monotonic()
    try:
        for name in ('jobs','runs','tiers','logs'):(base/name).mkdir()
        write(base/'config.json',dict(cfg,actual_project=str(root),actual_prior=str(prior),TMUX=os.environ['TMUX'],time=utc()))
        write(base/'CODE_FREEZE.json',freeze)
        approval=verify_stage02(prior);write(base/'STAGE0_2_APPROVAL_VERIFIED.json',approval)
        prior_hashes=hash_tree(prior);source_hashes=hash_tree(approval['source_code_path'])
        write(base/'PRIOR_SUCCESS_HASHES_BEFORE.json',dict(output=prior_hashes,source=source_hashes))
        old_inputs=verify_project(root,freeze['project_manifest_sha256']);write(base/'INPUT_HASHES_BEFORE.json',old_inputs)
        old_env=existing_environment_snapshot();write(base/'ENVIRONMENT_BEFORE.json',old_env)
        import importlib.metadata as metadata
        for name,version in cfg['expected_versions'].items():assert metadata.version(name)==version,'Package version changed: '+name
        gym_hashes=read(prior/'GYM_INPUT_HASHES.json')
        for p,h in gym_hashes.items():assert sha(p)==h
        gpu=idle_gpu(args.gpu);write(base/'GPU_BEFORE.json',dict(selected=gpu,nvidia_smi=full_smi()))
        initial=limits_snapshot();write(base/'RESOURCES_BEFORE.json',initial)
        assert shutil.disk_usage(base).free>=cfg['minimum_free_disk_bytes'],'Insufficient disk headroom'
        jobs={};paths={}
        for algorithm,levels in (('IQL',(1,2,4,6)),('BC',(1,))):
            for n in levels:
                tier='%s_n%d'%(algorithm,n);paths[tier]=[]
                for seed in range(901,901+n):
                    identity='%s_s%d'%(tier,seed)
                    job=dict(id=identity,tier=tier,algorithm=algorithm,concurrency=n,seed=seed,steps=20000,
                        task='hopper-morph',variant='BC_all',project=str(root),gpu_uuid=gpu['uuid'],
                        controller_pid=os.getpid(),barrier_timeout_seconds=cfg['tier_timeout_seconds'],formal_result=False)
                    p=base/'jobs'/(identity+'.json');write(p,job);paths[tier].append(p);jobs[identity]=sha(p)
        write(base/'WORKER_AUTHORIZATION.json',dict(scope='V12_0.3_benchmark_only',jobs=jobs,
            maximum_concurrency=6,maximum_IQL_jobs=13,maximum_BC_jobs=1,
            maximum_IQL_updates=260000,maximum_BC_updates=20000,later_stages_authorized=False))
        write(base/'PRE_RESULT_COMMITMENT.json',dict(config_sha256=sha(base/'config.json'),code_freeze_sha256=sha(HERE/'FROZEN.json'),
            jobs=jobs,resource_limits=cfg['resource_limits'],before_any_new_optimization=True,time=utc()))
        rss=None;gpu_per=None;stop_pressure=False;higher_IQL_blocked=False
        for tier in ('IQL_n1','IQL_n2','IQL_n4','IQL_n6','BC_n1'):
            n=len(paths[tier]);limits=limits_snapshot()
            reason=tier_allowed(n,limits,cfg['resource_limits'],rss,gpu_per,gpu['memory_total_MiB'])
            if tier.startswith('IQL') and higher_IQL_blocked:reason.append('A lower IQL tier was resource-limited; do not jump to a higher tier')
            if stop_pressure:reason.append('Persistent resource pressure in an earlier tier; stop remaining tests')
            if (base/'STOP_AFTER_TIER').exists():reason.append('User stop-after-tier marker exists')
            write(base/(tier+'.RESOURCE_DECISION.json'),dict(limits=limits,rejection_reasons=reason,time=utc()))
            if reason:
                if tier.startswith('IQL'):higher_IQL_blocked=True
                skipped.append(dict(tier=tier,reasons=reason));print('Skipping',tier,reason,flush=True);continue
            # Before starting a new tier, earlier workers and their CPU audits have all exited.
            write(base/(tier+'.GPU_ADMISSION.json'),idle_gpu(gpu['uuid']))
            metric,done=run_tier(base,tier,paths[tier],limits,cfg,gpu)
            measured.append(metric);completions.extend(done)
            rss=max(rss or 0,metric['max_worker_RSS_bytes'])
            gpu_per=max(gpu_per or 0,(metric['max_GPU_memory_used_MiB']-cfg['resource_admission']['max_resident_MiB'])/n)
            stop_pressure=stop_pressure or bool(metric['pressure_warnings'])
        analysis=analyze(measured);write(base/'THROUGHPUT_RESULT.json',dict(measured=measured,skipped=skipped,analysis=analysis))
        assert verify_project(root,freeze['project_manifest_sha256'])==old_inputs
        check_environment_protection(old_env)
        assert hash_tree(prior)==prior_hashes and hash_tree(approval['source_code_path'])==source_hashes
        for p,h in gym_hashes.items():assert sha(p)==h
        write(base/'INPUT_PROTECTION_AFTER.json',dict(passed=True,original_transfer_unchanged=True,prior_0_2_output_and_source_unchanged=True,
            Conda_metadata_and_original_MuJoCo_binding_unchanged=True,Gym_files_unchanged=True))
        report(base,measured,skipped,analysis)
        receipt=dict(completed=True,all_planned_tiers_measured=not bool(skipped),
            acceptance_status='AWAITING_USER_DECISION_RESOURCE_LIMITED' if skipped else 'AWAITING_USER_STAGE1_DECISION',
            measured_tiers=[x['name'] for x in measured],skipped_tiers=skipped,workers_completed_and_CPU_audited=len(completions),
            IQL_updates=sum(c['training_updates'] for c in completions if c['job']['algorithm']=='IQL'),
            BC_actor_updates=sum(c['training_updates'] for c in completions if c['job']['algorithm']=='BC'),
            Adam_updates=sum(c['counts']['Adam_updates'] for c in completions),
            environment_steps=sum(c['counts']['constructor_environment_steps']+c['counts']['evaluation_environment_steps'] for c in completions),
            evaluation_episodes=10*len(completions),recommended_measured_concurrency=analysis['recommended_measured_concurrency'],
            physical_GPU_UUID=gpu['uuid'],later_stages_started=False,formal_results=False,
            elapsed_seconds=time.monotonic()-started,ended_utc=utc())
        receipt['evidence_sha256']={p.relative_to(base).as_posix():sha(p) for p in base.rglob('*')
            if p.is_file() and 'private_runtime' not in p.relative_to(base).parts and 'cuda_cache' not in p.relative_to(base).parts}
        write(base/'STAGE0_3_COMPLETED.json',receipt)
        print(json.dumps({k:v for k,v in receipt.items() if k!='evidence_sha256'},ensure_ascii=False,indent=2),flush=True)
        print('Stopped. Read',base/'REPORT_ZH.md',flush=True);return 0
    except BaseException:
        error=traceback.format_exc();protection={}
        try:
            if old_inputs is not None:protection['transfer_unchanged']=verify_project(root,freeze['project_manifest_sha256'])==old_inputs
            if old_env is not None:check_environment_protection(old_env);protection['environment_unchanged']=True
            if prior_hashes is not None:protection['prior_0_2_unchanged']=hash_tree(prior)==prior_hashes
        except BaseException:protection['check_error']=traceback.format_exc()
        write(base/'FAILED.json',dict(time=utc(),traceback=error,measured_tiers=[m['name'] for m in measured],
            input_protection=protection,automatic_retry=False,later_stages_started=False))
        if not (base/'REPORT_ZH.md').exists():report(base,measured,skipped,analyze(measured),error)
        print(error,file=sys.stderr,flush=True);return 1


if __name__=='__main__':raise SystemExit(main())
