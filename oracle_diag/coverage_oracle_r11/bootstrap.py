"""Prepare an isolated R11 stage. Copies read-only R09 science, narrows dispatch."""
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD_CODE=ROOT/'oracle_diag/coverage_oracle_r09'
OLD_OUT=ROOT/'oracle_diag/results/coverage_oracle_20261016_r09'
OUT=ROOT/'oracle_diag/results/coverage_oracle_20261016_r11'
DOC=ROOT/'outputs/research_idea_protocol_20261016_r11'

def read(path):return json.loads(Path(path).read_text(encoding='utf-8-sig'))
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    with open(path,'x',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)

def main():
    OUT.mkdir(parents=True,exist_ok=False)
    DOC.mkdir(parents=True,exist_ok=False)
    copied={}
    for name in ('learner.py','train.py','audit.py','selection.py'):
        original=OLD_CODE/name
        target=HERE/name
        with open(target,'x',encoding='utf-8',newline='') as f:f.write(original.read_text(encoding='utf-8'))
        copied[name]=dict(from_path=str(original),from_sha256=sha(original),to_sha256=sha(target))
        assert copied[name]['from_sha256']==copied[name]['to_sha256']
    rt=(OLD_CODE/'runtime.py').read_text(encoding='utf-8')
    rt=rt.replace('Isolated R09 runtime.','Isolated R11 runtime.').replace('coverage_oracle_20261016_r09','coverage_oracle_20261016_r11')
    rt=rt.replace('research_idea_protocol_20261016_r09','research_idea_protocol_20261016_r11')
    start=rt.index('def check_frozen():')
    end=rt.index('def configure_cpu():')
    rt=rt[:start]+'''def check_frozen():
    a=read(OUT/'stage_authorization.json')
    assert a['scope']=='T12_reference_diagnosis_scores_five_smokes_only'
    assert a['protocol_sha256']==sha(OUT/'protocol.json')
    assert all(a[k] is False for k in ('U_validation_authorized','G_validation_authorized',
        'C_grid_authorized','formal_authorized','task2_authorized','Hopper_cache_authorized'))
    for path,expected in read(OUT/'code_frozen.json')['sha256'].items():
        assert sha(path)==expected,'Code changed after freeze: '+path

def check_job(job,for_launch=True):
    check_frozen()
    assert job['task']=='walker2d-morph'
    arm,phase,s,h=(job[k] for k in ('arm','phase','seed','steps'))
    assert job['temp'] in (1.,3.,10.)
    if phase=='tune':
        assert arm=='T' and s in (101,102,103) and h in (500000,1000000)
        if h==1000000:
            chosen=read(OUT/'TEMP_500K_SELECTED.json')
            assert job['temp']==chosen['temp'],'1M temperature must stay locked to 500k winner'
    elif phase=='reference':
        assert arm=='T' and s==1001
        chosen=read(OUT/'T_selected.json')
        assert job['temp']==chosen['temp'] and h==chosen['steps']
    elif phase=='smoke':
        assert arm in ('G','C','R','U','D') and s==7001 and h==5000
        chosen=read(OUT/'T_selected.json')
        assert job['temp']==chosen['temp']
        assert read(OUT/'scores/completed.json')['passed']
        if arm in ('G','C','R'):assert job['ratio']==.2
    else:
        raise AssertionError('Not authorized in R11: '+phase)
    assert read(OUT/'cpu_acceptance.json')['passed']
    assert read(OUT/'T_smoke_audited.json')['passed']
    assert OUT in Path(job['out']).resolve().parents
    if for_launch:
        assert not (OUT/'STAGE_COMPLETED.json').exists(),'Stage ended; separate U/G authorization required'

'''+rt[end:]
    with open(HERE/'runtime.py','x',encoding='utf-8',newline='') as f:f.write(rt)
    p=read(ROOT/'outputs/research_idea_protocol_20261016_r10/PROTOCOL_REVISION_R10.json')
    p['version']='R11'
    p['status']='AUTHORIZED_T12_REFERENCE_DIAGNOSIS_SCORES_FIVE_SMOKES_ONLY'
    p['execution_authorized']='current user segmented authorization; no U/G validations'
    p['recommended_defaults']['T_H_adoption']['window_500k']=[460000,470000,480000,490000,500000]
    p['recommended_defaults']['T_H_adoption']['window_1M']=[960000,970000,980000,990000,1000000]
    p['recommended_defaults']['T_H_adoption']['comparison_limit']='different late-training stages; not matched training stages or a common absolute-step window; cosine schedules differ'
    p['recommended_defaults']['new_code_path_proposal']=str(HERE)
    p['recommended_defaults']['new_result_path_proposal']=str(OUT)
    p['stage_scope']=dict(allowed=['T lean12 including already audited seed101/temp1/500k',
        'reference T1','CPU diagnosis5000','zero-additional-training score audit',
        'remaining G/C/R/U/D one5k smoke each'],
        long_runs=13,smoke_runs_including_reused_T=6,total_jobs=19,
        already_complete_long=1,already_complete_smoke=1,new_long_runs=12,new_smoke_runs=5,
        prohibited=['U validation3','G validation9','C grid','formal','task2','Hopper cache','old queue resume'],
        mandatory_stop='after five remaining smokes pass; await separate U/G authorization',
        T_locked_report='deliver after tuning; already authorized reference/diagnosis/smokes may continue')
    write(OUT/'protocol.json',p)
    write(DOC/'PROTOCOL_R11.json',p)
    write(OUT/'stage_authorization.json',dict(scope='T12_reference_diagnosis_scores_five_smokes_only',
        protocol_sha256=sha(OUT/'protocol.json'),U_validation_authorized=False,G_validation_authorized=False,
        C_grid_authorized=False,formal_authorized=False,task2_authorized=False,Hopper_cache_authorized=False,
        user_evidence='2026-10-16：本轮只授权T调参12次（含seed101/temp1）、参考T1次、5000步诊断、零新增训练分数审计，以及其余五组各1次5k冒烟；U3/G9待T锁定后单独授权。',
        prior_hold='no new jobs dispatched during R10 budget hold; first common T finished and CPU-audited'))
    write(OUT/'science_copy_provenance.json',dict(copied_unchanged=copied,
        runtime_changes='new output root and strictly narrower job guard; unchanged data, seeds, learner, evaluator and selection math',
        old_runtime_sha256=sha(OLD_CODE/'runtime.py')))
    write(OUT/'active_execution.json',dict(code=str(HERE),results=str(OUT),queue_script=str(HERE/'pipeline.py'),
        queue_status=str(OUT/'queue_status.json'),protocol_sha256=sha(OUT/'protocol.json'),
        scope='no U/G validation, only U/G 5k smoke',reused_R09=str(OLD_OUT),
        note='old R09 queue_status is historical; do not resume it'))
    contract='''# R11 分段执行合同

用户已接受R10计数与耗时，并进一步缩小本轮执行范围。本轮只完成T配置、固定参考、诊断、分数审计和剩余五组冒烟。

## 规则确认

- 500k：temp1/3/10各seed101–103，9次；只按500k三seed Last-5均值选temp，平局沿用3/1/10。
- 1M：仅500k胜出temp，种子仍101–103，3次，从零训练；不额外搜索温度。
- 同seed差d=Last5(1M)−Last5(500k)，仅配对均值严格>2×sample_SD(d)/sqrt(3)时采用1M，否则500k。该规则有意偏向短H，不能解释为等效或严格95%结论。
- 两档Last-5窗口分别460k–500k、960k–1M：各取末期五点，但属于不同训练阶段，不能写成相同训练阶段/相同绝对步数窗口的比较。两档cosine周期也不同。
- [3,7]未来T128描述性对照规则沿用，本轮没有正式阶段，不执行。

## 当前事实与复用

旧清单只有T冒烟与首个T，是历史派发记录。旧队列56268已停止，预算等待期间没有新增派发；首个temp1/seed101/500k已完成并CPU审计通过。它计入12次T，不重跑；T冒烟也复用。不改变这两项旧结果。

## 本轮边界

T12＋参考T1＝13项长训练；T/G/C/R/U/D六项冒烟，其中T已完成。实际新增12项长训练＋5项冒烟，共17个新GPU作业。之后固定参考下CPU采5000步诊断、零新增训练审计分数，保持R09既定公式、种子、采样、source通道和解释限定。

T锁定后交付温度/H、逐seed两档窗口、配对差/SD/SE和规则判定；参考/诊断/五组冒烟可在本次授权内继续。五组冒烟后必须停止。**U的3次验证、G的9次比例验证、C比例网格、正式阶段、第二任务与Hopper缓存均未授权。** U/G各一次5k管线冒烟不等于其验证训练。

## 验收与保留

新代码和结果放R11新目录；学习器/训练/审计/选择代码从冻结R09逐字节复制并核对hash。只改独立输出根、派发范围与调度/H选择；先用CPU补验这些变动，不重跑旧CPU代数验收或已完成训练。原保护清单延续并加上R09已完成产物。单GPU串行、隐藏窗口、诊断CPU，不改旧代码结果、不终止其他应用。异常保留原始目录，不自动重跑。
'''
    with open(DOC/'STAGE_CONTRACT_R11.md','x',encoding='utf-8') as f:f.write(contract)
    with open(DOC/'WORKING_IDEA_CANVAS_R11.md','x',encoding='utf-8') as f:
        f.write('''# Working Idea Canvas R11

|项目|状态|
|---|---|
|研究主张|PROVISIONAL：保持R09/R10，未把单种子验证回报变成研究结论|
|T搜索|USER CONFIRMED：500k三温度各三seed；胜出温度1M只跑同三seed|
|H比较|USER CONFIRMED：不同末期窗口配对，均值>2SE才取1M，有意偏向短H；不是相同阶段比较|
|当前证据|VERIFIED：首项T与T冒烟已完成并审计；预算等待时没有新增派发|
|授权|USER CONFIRMED：只T12/参考1/诊断5000/零训练分布审计/五组冒烟|
|U/G验证|UNAUTHORIZED：T配置锁定后仍需单独授权；本队列没有验证派发路径|
|Hopper|OUT OF SCOPE：未来第二任务授权后、其GPU前全量CPU缓存验收|
|下一交付|T锁定报告及分布/冒烟报告，之后等待单独授权|
''')
    print(str(OUT),flush=True)

if __name__=='__main__':main()
