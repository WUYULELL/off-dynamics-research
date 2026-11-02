# -*- coding: utf-8 -*-
"""Create a fresh Stage0-only recovery; leave all previous frozen files intact."""
import ast,json,time,hashlib,shutil,difflib
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'oracle_diag/v11_source_actor'
CANDIDATE=ROOT/'oracle_diag/v11_source_actor_recovery_20261102'
HERE=Path(__file__).resolve().parent
OLD=ROOT/'outputs/v11_source_actor_20261101/stage0'
PRIOR=ROOT/'outputs/v11_source_actor_20261102_recovery1/stage0'
OUT=ROOT/'outputs/v11_source_actor_20261102_stage0_recovery2/stage0'
DOCUMENT=Path('C:/Users/TXin/Desktop/Claude命令/给Code的指令_v11修订版_源数据动作进入策略的上限检验.md')

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    with open(p,'x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)
def textnew(p,s):
    with open(p,'x',encoding='utf-8',newline='\n') as f:f.write(s)

def main():
    assert not OUT.exists()
    OUT.mkdir(parents=True);(OUT/'logs').mkdir();(OUT/'runs').mkdir()
    for name in ('actor.py','cpu_acceptance.py'):shutil.copyfile(BASE/name,HERE/name)
    common=(CANDIDATE/'common.py').read_text(encoding='utf-8')
    common=common.replace('V11 Stage0 then the authorized Hopper Stage1; unchanged O1 dependencies.','V11 revised: authorized Stage0 only, then stop; unchanged O1 science.')
    common=common.replace('outputs/v11_source_actor_20261102_recovery1/stage0','outputs/v11_source_actor_20261102_stage0_recovery2/stage0')
    common=common.replace("'V11_Stage0_then_Hopper_Stage1_then_stop'","'V11_revision_Stage0_only_then_stop'")
    common=common.replace("assert c['stage1_authorized'] and not c['stage2_authorized']","assert not c['stage1_authorized'] and not c['stage2_authorized']")
    common=common.replace("        paths = list(WALKER.rglob('*'))","        paths = list(WALKER.rglob('*'))\n        for stable in ('oracle_diag/v11_source_actor','oracle_diag/v11_source_actor_recovery_20261102',\n                       'outputs/v11_source_actor_20261101','outputs/v11_source_actor_20261102_recovery1'):\n            paths += list((ROOT/stable).rglob('*'))")
    tree=ast.parse(common);fn=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='formal_jobs')
    lines=common.splitlines(True);lines[fn.lineno-1:fn.end_lineno]=["def formal_jobs():\n    raise PermissionError('Stages1/2 require explicit confirmation after Stage0 reporting')\n"]
    textnew(HERE/'common.py',''.join(lines))
    trial=(CANDIDATE/'trial.py').read_text(encoding='utf-8')
    trial=trial.replace('Whitelisted Stage0 diagnostics and Stage1 formal jobs; single-GPU controller only.','Authorized Stage0 diagnostics only; no formal training entry point.')
    trial=trial.replace("    guard(); assert r.read(OUT/'CPU_ACCEPTANCE.json')['passed']; r.configure_cpu()","    guard(); assert not formal, 'Stage1 is not authorized'; assert r.read(OUT/'CPU_ACCEPTANCE.json')['passed']; r.configure_cpu()")
    trial=trial.replace("parser.add_argument('--trial',required=True);parser.add_argument('--formal',action='store_true')","parser.add_argument('--trial',choices=[v[0] for v in TRIALS],required=True)")
    trial=trial.replace('args=parser.parse_args();main(args.trial,args.formal)','args=parser.parse_args();main(args.trial,False)')
    textnew(HERE/'trial.py',trial)
    audit=(CANDIDATE/'audit.py').read_text(encoding='utf-8')
    audit=audit.replace('accum={};curves=[];block={};count=0','accum={};curves=[];block={};count=0;zero_selected_batches=0')
    audit=audit.replace('            for k in keys:\n',"            if wa and row['source_selected']==0:zero_selected_batches+=1\n            for k in keys:\n")
    audit=audit.replace('curve_every1000=curves,all_steps_source_weight_and_MSE_recomputed=bool(wa))','curve_every1000=curves,all_steps_source_weight_and_MSE_recomputed=bool(wa),\n        zero_selected_batches=zero_selected_batches,zero_selected_batch_fraction=zero_selected_batches/count if wa else None)')
    textnew(HERE/'audit.py',audit)
    verify=(CANDIDATE/'verify.py').read_text(encoding='utf-8').replace('Final Stage0 CPU evidence/receipt verification before automatic Stage1.','Final Stage0 CPU verification; later stages are unauthorized.')
    verify=verify.replace("and settings()['stage1_authorized']","and not settings()['stage1_authorized'] and not settings()['stage2_authorized']")
    verify=verify.replace("    assert not torch.cuda.is_initialized()\n    r.write(OUT/'DELIVERY_VERIFIED.json'","    for p,h in r.read(OUT/'REPAIR_CHANGELOG.json')['evidence_sha256'].items():assert r.sha(p)==h,p\n    assert not torch.cuda.is_initialized()\n    r.write(OUT/'DELIVERY_VERIFIED.json'")
    textnew(HERE/'verify.py',verify)
    report=(CANDIDATE/'report.py').read_text(encoding='utf-8')
    report=report.replace("correction=settings()['correction'],automatic_authorized_Stage1_next=True,time=time.time())","correction=settings()['correction'],automatic_authorized_Stage1_next=False,\n        stage1_authorized=False,stage2_authorized=False,repair_evidence=r.read(OUT/'REPAIR_CHANGELOG.json'),time=time.time())")
    report=report.replace('按v11表格对入选样本求均值','按用户最新澄清对入选样本求均值（分母=入选数）')
    report=report.replace('## 后续预算与接续','## 后续预算（仅估计，未授权启动）')
    report=report.replace('按用户最新授权，阶段0核验通过后直接运行阶段1，SA-all201–210后SA-top201–210，阶段1完成停下。阶段2Walker上限、V9阶段4、T调参均不启动。','本次按v11修订版只执行阶段0，完成后队列退出并停下汇报。阶段1/阶段2的任务、变体和先后顺序均须用户之后明确确认。Walker后续BC参照确定为BC-all均值38.85（精确值和收据见config.json）；V9阶段4和T调参未授权。')
    needle="    with open(r.winpath(OUT/'STAGE0_REPORT_ZH.md'),'x',encoding='utf-8') as f:f.write('\\n'.join(lines))"
    assert needle in report
    extra="""    lines+=['','## 无入选样本的边界情况','',
        'SA-top源项=sum(入选exp(10A)×动作MSE)/入选数，再乘w_a。无人入选时源项0，不除以0。']
    for key in ('SA_all_s201','SA_top_s201'):
        d=results[key]['diagnostics']
        lines.append('- %s：无人入选批次%d/%d，占比%.6f。'%(key,d['zero_selected_batches'],d['steps'],d['zero_selected_batch_fraction']))
    changes=r.read(OUT/'REPAIR_CHANGELOG.json')
    lines+=['','## 修复文件、行号与一致性证据','',
        '故障修复本身仅明确O1评估函数的导入路径。另列修订版要求的停止范围、配置/报告/诊断改动。',
        '|用途|文件|本次行号|说明|','|---|---|---|---|']
    for item in changes['items']:
        lines.append('|%s|%s|%s|%s|'%(item['category'],item['file'],','.join(map(str,item['new_lines'])),item['description']))
    lines+=['',
        '训练逻辑：actor.py逐字节等同原CPU验收版本；短试跑训练循环AST、V/Q、三Adam和cosine更新顺序与候选相同。SA-top分母按用户澄清维持候选，非修复时暗改。',
        '随机序列：初始化与Sampler源码未改；同seed初始化和完整target/source索引流与O1独立重建值匹配，actor不额外抽样。',
        '评估协议：明确文件路径加载原冻结O1同一evaluate，函数AST/文件SHA核验；每10k十episode、固定evalseed、shifted XML、.002/1000、原归一化、保存恢复Python/NumPy/Torch/CUDA训练RNG均沿用。',
        '本次额外CPU优化0，复用此前8次合成learner更新/24Adam。原启动失败发生在trial.main之前，原GPU更新/环境步0。旧文件和失败现场保留，证据哈希见REPAIR_CHANGELOG.json及DELIVERY_VERIFIED.json。','']
"""
    report=report.replace(needle,extra+needle)
    textnew(HERE/'report.py',report)
    for p in HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'),filename=str(p))
    shutil.copyfile(DOCUMENT,OUT/'INSTRUCTION_V11_REVISED.md')
    config=read(OLD/'config.json')
    config.update(scope='V11_revision_Stage0_only_then_stop',stage1_authorized=False,stage2_authorized=False,
        original_instruction_path=str(DOCUMENT),original_instruction_sha256=sha(DOCUMENT),instruction_version='v11 revised supersedes original',
        source_top_reduction='sum_selected(exp(10A)*action_dim_mean_MSE)/selected_count; multiply w_a; selected_count0 =>0',
        source_top_denominator='selected_count, NOT64',source_top_empty_batch_fraction_required=True,
        all_later_stage_order_requires_new_user_confirmation=True)
    result_path=ROOT/'outputs/v9_hopper_label_headroom_20261101/walker_stage3/WALKER_RESULT.json'
    walker=read(result_path)
    config['future_Walker_BC_reference']=dict(group='BC_all',display_mean=38.85,
        exact_mean=walker['statistics']['Last5']['groups']['BC_all']['mean'],seeds=list(range(301,306)),
        source=str(result_path),source_sha256=sha(result_path),future_experiment_authorized=False)
    write(OUT/'config.json',config)
    write(OUT/'authorization.json',dict(scope=config['scope'],stage0_recovery_authorized=True,
        stage1_authorized=False,stage2_authorized=False,old_automatic_Stage1_authorization_revoked=True,
        source='Latest explicit user confirmation: prepared import fix, revised Stage0 ONLY, stop and report; later task order requires confirmation',
        denominator_clarification='SA-top divide by selected_count, empty0, record empty batch fraction',
        no_automatic_recovery_on_any_additional_failure=True,time=time.time()))
    shutil.copyfile(OLD/'CPU_ACCEPTANCE.json',OUT/'CPU_ACCEPTANCE.json')
    write(OUT/'CPU_ACCEPTANCE_REUSE.json',dict(passed=True,original_receipt=str(OLD/'CPU_ACCEPTANCE.json'),
        original_sha256=sha(OLD/'CPU_ACCEPTANCE.json'),new_synthetic_optimizer_updates=0,
        total_previously_accepted_synthetic_learner_updates=8,total_previously_accepted_Adam_steps=24,
        actor_py_byte_identical=sha(BASE/'actor.py')==sha(HERE/'actor.py'),
        learner_config_identical=read(OLD/'config.json')['learner']==config['learner']))
    write(OUT/'BUDGET_ESTIMATE.json',dict(stage0_minutes_estimate=[20,35],
        historical_O1_seed201_500k_seconds=8067.8929278850555,
        Stage0_GPU_new_updates=50000,Stage0_GPU_reference_updates=10000,total_GPU_learner_updates=60000,
        Stage0_GPU_Adam_steps=180000,evaluation_episodes=50,new_CPU_optimizer_updates=0,
        original_CPU_synthetic_learner_updates_reused=8,single500k_prospective_budget_measured_after_trials=True,
        stages1_and2_not_authorized=True))
    write(OUT/'RECOVERY_PLAN.json',dict(scope=config['scope'],source_candidate=str(PRIOR),
        root_cause='ambiguous train import resolved repository-root file',
        approved_repair='Explicit importlib file path to unchanged O1 stage2/train.py evaluate',
        stop_after_Stage0=True,stage1_authorized=False,stage2_authorized=False,code=str(HERE),output=str(OUT),
        original_GPU_updates=0,original_environment_steps=0,new_CPU_updates=0,reuse_CPU_synthetic_updates=8,time=time.time()))
    write(OUT/'RECOVERY_AUTHORIZED.json',dict(authorized=True,scope=config['scope'],
        recovery_plan_sha256=sha(OUT/'RECOVERY_PLAN.json'),explicit_user_confirmation_received=True,
        stop_after_Stage0=True,stage1_authorized=False,stage2_authorized=False,time=time.time()))
    print(json.dumps(dict(prepared=True,code=str(HERE),output=str(OUT),actor_byte_identical=sha(HERE/'actor.py')==sha(BASE/'actor.py'),later_stages_authorized=False)))

if __name__=='__main__':main()
