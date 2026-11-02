# -*- coding: utf-8 -*-
"""Prepare the user's explicitly requested recovery; never launch training."""
import ast
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD_CODE = ROOT/'oracle_diag/coverage_raw_r14'
OLD = ROOT/'oracle_diag/results/coverage_raw_20261026_r14'
OLD_DOC = ROOT/'outputs/research_idea_protocol_20261026_r14'
OUT = ROOT/'oracle_diag/results/coverage_raw_20261029_r14_finish1'
DOC = ROOT/'outputs/research_idea_protocol_20261029_r14_finish1'
sys.path.insert(0, str(OLD_CODE))
from runtime import read, sha, write, check_frozen

def main():
    check_frozen()
    if OUT.exists():
        assert {p.name for p in OUT.iterdir()} <= {'jobs','logs','PREPARATION_FAILURE.json'}
        assert not any((OUT/'jobs').iterdir()) and not any((OUT/'logs').iterdir())
    if DOC.exists(): assert not any(DOC.iterdir())
    OUT.mkdir(exist_ok=True); DOC.mkdir(exist_ok=True)
    for n in ('jobs','logs'): (OUT/n).mkdir(exist_ok=True)
    q = read(OLD/'queue_status.json')
    assert q['stage']=='FAILED' and q['audited_new_jobs']==33
    query = "Get-CimInstance Win32_Process | Where-Object { $_.Name -eq 'python.exe' -and $_.CommandLine -match 'coverage_raw_r14.*(pipeline|train|audit)\\.py' } | Select-Object ProcessId,ParentProcessId,CreationDate,CommandLine | ConvertTo-Json -Depth 3"
    powershell = str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/native/powershell/pwsh.exe')
    procs = subprocess.check_output([powershell,'-NoProfile','-Command',query],encoding='utf-8-sig')
    assert not procs.strip(), procs
    event_query = "Get-WinEvent -FilterHashtable @{LogName='System'; Id=1074,6005,6006,6008,41; StartTime=[datetime]'2026-10-29 20:05:00'; EndTime=[datetime]'2026-10-29 20:20:00'} -ErrorAction SilentlyContinue | Select-Object TimeCreated,Id,ProviderName,Message | ConvertTo-Json -Depth 3"
    events = subprocess.check_output([powershell,'-NoProfile','-Command',event_query],encoding='utf-8-sig')
    interrupted = OLD/'runs/hopp_w1_s204'
    progress = read(interrupted/'progress.json')
    assert progress['step']==3000 and progress['finite']
    assert not list(interrupted.glob('model_*.pt'))
    assert not (interrupted/'completed.json').exists()
    evidence_paths = list(interrupted.rglob('*')) + [OLD/'queue_status.json',Path(q['failure']),OLD/'launch1.stderr.log',OLD/'logs/hopp_w1_s204.stderr.log']
    evidence = {str(p):sha(p) for p in evidence_paths if p.is_file()}
    write(OUT/'INTERRUPTION_DIAGNOSIS.json',dict(CPU_only=True,time=time.time(),
        no_training_processes=True,system_events=json.loads(events),queue_failure=read(q['failure']),
        interrupted_progress=progress,checkpoint_files=[],raw_evaluation_events=0,
        conclusion='OS restart/termination sequence; no saved checkpoint; explicit user request authorizes a fresh attempt of the same logical run',
        cause_scope='System records a restart request and event-log shutdown/start; no inference about the initiating human or software intent',
        evidence_sha256=evidence,no_GPU_or_environment_steps=True))
    protected = dict(read(OLD/'protected_manifest.json')['sha256'])
    for p,h in protected.items(): assert sha(p)==h, 'Previously protected file changed: '+p
    for directory in (OLD,OLD_DOC,OLD_CODE):
        for p in directory.rglob('*'):
            if p.is_file() and '__pycache__' not in str(p):
                h=sha(p)
                if str(p) in protected: assert protected[str(p)]==h
                protected[str(p)]=h
    write(OUT/'protected_manifest.json',dict(count=len(protected),sha256=protected,no_old_writes=True))
    for n in ('protocol.json','PRE_RESULT_COMMITMENT.json'):
        (OUT/n).write_bytes((OLD/n).read_bytes())
    (DOC/'STAGE_CONTRACT_R14.md').write_bytes((OLD_DOC/'STAGE_CONTRACT_R14.md').read_bytes())
    (DOC/'BUDGET_ESTIMATE.json').write_bytes((OLD_DOC/'BUDGET_ESTIMATE.json').read_bytes())
    original_plan=read(OLD/'training_plan.json')['jobs']; reused=list(read(OLD/'reuse_receipts.json')['runs'])
    for r in reused:
        assert sha(r['audit'])==r['audit_sha256'] and read(r['audit'])['passed']
        for p,h in r['files_sha256'].items(): assert sha(p)==h
    complete=[]; remaining=[]
    for j in original_plan:
        p=Path(j['out'])
        if (p/'completed.json').exists():
            a=read(p/'audit.json'); done=read(p/'completed.json')
            assert a['passed'] and done['steps']==500000 and done['finite']
            assert sha(done['final_model'])==done['final_model_sha256']==a['final_model_sha256']
            assert a['completed_sha256']==sha(p/'completed.json') and a['evaluations_sha256']==sha(p/'evaluations.json')
            files={str(f):sha(f) for f in p.rglob('*') if f.is_file() and '__pycache__' not in str(f)}
            reused.append(dict(task=j['task'],seed=j['seed'],w=j['source_w'],label='raw',out=str(p),audit=str(p/'audit.json'),audit_sha256=sha(p/'audit.json'),files_sha256=files))
            complete.append(j['id'])
        else:
            new=dict(j,out=str(OUT/'runs'/j['id'])); remaining.append(new)
            write(OUT/'jobs'/(j['id']+'.json'),new)
    assert len(complete)==33 and len(remaining)==7 and len(reused)==43
    assert [j['id'] for j in remaining]==['hopp_w1_s%d'%s for s in range(204,211)]
    write(OUT/'reuse_receipts.json',dict(runs=reused,completed_R14_runs=33,prior_R13_runs=10,no_completed_run_retrained=True))
    write(OUT/'training_plan.json',dict(jobs=remaining,new_GPU_jobs=7,new_updates=3500000,
        logical_R14_jobs=40,reused_R14_completed_jobs=33,no_next_stage=True))
    write(OUT/'recovery.json',dict(time=time.time(),explicit_user_resume=True,original_results=str(OLD),
        completed_reused=complete,reissued_logical_ids=[j['id'] for j in remaining],
        restart_from_zero_reason='hopp_w1_s204 stopped at step3000 without any saved model checkpoint',
        last_observed_unsaved_updates=3000,exact_discarded_updates_unknown=True,
        other_six_jobs_previously_unstarted=True,logical_total_unchanged=40,
        no_checkpoint_restore_or_GPU_loss_overlap_claim=True,old_failure_preserved=str(q['failure'])))
    # Byte-identical science, data loading, evaluation and reporting. Only output roots differ.
    for n in ('learner.py','train.py','audit.py','metrics.py','analyze.py'):
        (HERE/n).write_bytes((OLD_CODE/n).read_bytes())
    runtime=(OLD_CODE/'runtime.py').read_text(encoding='utf-8')
    runtime=runtime.replace("OUT = ROOT / 'oracle_diag/results/coverage_raw_20261026_r14'", "OUT = ROOT / 'oracle_diag/results/coverage_raw_20261029_r14_finish1'")
    runtime=runtime.replace("DOC = ROOT / 'outputs/research_idea_protocol_20261026_r14'", "DOC = ROOT / 'outputs/research_idea_protocol_20261029_r14_finish1'")
    (HERE/'runtime.py').write_text(runtime,encoding='utf-8')
    old_ast=ast.parse((OLD_CODE/'runtime.py').read_text(encoding='utf-8')); new_ast=ast.parse(runtime)
    assert {n.name:ast.dump(n) for n in old_ast.body if isinstance(n,ast.FunctionDef)}=={n.name:ast.dump(n) for n in new_ast.body if isinstance(n,ast.FunctionDef)}
    pipeline=(OLD_CODE/'pipeline.py').read_text(encoding='utf-8')
    pipeline=pipeline.replace("w=0.,audit=a,out=r['out']", "w=r['w'],audit=a,out=r['out']")
    pipeline=pipeline.replace('assert len(self.results)==5','assert len(self.results)==38')
    pipeline=pipeline.replace('dispatched_new_jobs=len(self.dispatched)','dispatched_new_jobs=33+len(self.dispatched), recovery_dispatched_jobs=len(self.dispatched)')
    pipeline=pipeline.replace('assert len(jobs)==40','assert len(jobs)==7')
    pipeline=pipeline.replace('new_GPU_jobs=40,all_audited=True','new_GPU_jobs=40,recovery_GPU_jobs=7,reused_R14_jobs=33,all_audited=True')
    (HERE/'pipeline.py').write_text(pipeline,encoding='utf-8')
    write(OUT/'engineering_delta.json',dict(science_files_byte_identical=['learner.py','train.py','audit.py','metrics.py','analyze.py'],runtime_function_AST_identical=True,
        runtime_changes='OUT/DOC roots only',pipeline_changes='reuse33 audited logical jobs; launch7 remaining; same final50 raw-input analysis',
        no_scientific_change=True,no_old_file_modified=True))
    write(OUT/'stage_authorization.json',dict(scope='R14_raw_source_strength_precheck_40_only',
        protocol_sha256=sha(OUT/'protocol.json'),new_GPU_jobs=40,recovery_GPU_jobs=7,
        estimator_development_authorized=False,other_experiments_authorized=False,
        explicit_user_recovery_request='2026-10-29: my experiment may have been interrupted by restart; please restart/continue',time=time.time()))
    scope='''# R14 重启后恢复范围\n\n用户2026-10-29明确要求继续实验。原队列已退出，保留原失败日志、原33项完成审计及中断产物。Windows事件记录重启请求、日志服务停止及重新启动；不推断重启发起者的意图。\n\n仅复用33项已审计R14结果；最后7项为Hopper w=1 seeds204–210，各500k/temp10。seed204中断时最后记录3000步，无任何模型检查点、无评估事件，只能在新目录用原seed从0重启该逻辑项；实际丢失更新数未必恰为3000。其余6项尚未开始。逻辑仍为原40项，不追加种子、强度或任务。\n\nlearner/train/audit/metrics/analyze逐字节原R14，runtime函数AST完全一致，仅变更输出路径。原protocol与结果前承诺逐字节保留，不根据已完成结果改变分析。新GPU作业7个、350万更新；保留全部旧文件哈希。原R14及R09/R11/R12/R13旧队列均不得重启。\n\n本次为从0新attempt，不声称恢复了3000步状态、优化器或通过GPU损失重叠。新attempt的3000步双抽样流须与中断记录相同；初始化须与旧attempt及同seed Hopper T一致。source标签/对齐、全部3200万双流、有限状态、TB/episode、XML/归一化仍由原审计检查。\n\n预计剩余19–22小时（持续单GPU），含启动前CPU检查、训练及最终审计报告。全部40项完成后独立从50分析记录的原始TB/episode按原协议报告并停止；不开发估计器，不追加训练。\n'''
    (DOC/'RECOVERY_SCOPE.md').write_text(scope,encoding='utf-8')
    write(OUT/'active_execution.json',dict(code=str(HERE),results=str(OUT),docs=str(DOC),queue_script=str(HERE/'pipeline.py'),
        original_queues_must_not_restart=True,logical_R14_jobs=40,reused_R14_completed=33,recovery_GPU_jobs=7))
    deps=list(read(OLD/'dependency_hashes.json')['sha256'])+[str(OLD/'code_frozen.json'),str(OLD/'cpu_acceptance.json'),str(OLD/'CPU_ACCEPTANCE_COMMITMENT.json'),str(OUT/'protocol.json'),str(OUT/'training_plan.json'),str(OUT/'reuse_receipts.json'),str(OUT/'stage_authorization.json'),str(DOC/'RECOVERY_SCOPE.md'),str(DOC/'STAGE_CONTRACT_R14.md')]+[str(p) for p in (OUT/'jobs').glob('*.json')]
    write(OUT/'dependency_hashes.json',dict(sha256={p:sha(p) for p in deps}))
    code=list(HERE.glob('*.py'))+[ROOT/'algo/offline_offline'/n for n in ('iql.py','iql_dyn.py','iql_unc.py')]
    write(OUT/'code_frozen.json',dict(sha256={str(p):sha(p) for p in code},science_unchanged=True))
    print(json.dumps(dict(prepared=True,protected=len(protected),reused=33,remaining=7,diagnosis=str(OUT/'INTERRUPTION_DIAGNOSIS.json'))))

if __name__=='__main__': main()
