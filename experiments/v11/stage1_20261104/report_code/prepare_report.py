# -*- coding: utf-8 -*-
from report_runtime import *
import shutil,ast,difflib
OUT=REPORT;OLD=base.OUT
def copy(source,target):
    assert not target.exists();target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(base.r.winpath(source),base.r.winpath(target))
for name in ('config.json','authorization.json','BUDGET_ESTIMATE.json','launch1.json','FROZEN.json','PRE_RESULT_COMMITMENT.json','SEED207_PREFIX_COMPARISON.json'):
    copy(OLD/name,OUT/('training_'+name) if name in ('FROZEN.json','PRE_RESULT_COMMITMENT.json') else OUT/name)
for name in ('training_plan.json','reference_receipts.json','protocol.json'):copy(OLD/'stage1'/name,OUT/'stage1'/name)
for name in ('stage1_report.py','stage1_verify.py'):
    s=(BASE_CODE/name).read_text(encoding='utf-8-sig')
    if name=='stage1_report.py':
        assert s.count("correction=settings()['correction']")==1
        s=s.replace("correction=settings()['correction']","correction=settings()['actor_loss']")
        s=s.replace("str(comparison['all_compared_diagnostics_bitwise_equal']),str(comparison['first_mismatch_step'])", "str(comparison['logs']['parsed_numeric_bits_equal']),str(comparison['logs']['first_mismatch'])")
    target=HERE/(name.replace('.py','_repaired.py'))
    with open(target,'x',encoding='utf-8') as f:f.write(s)
    patch=''.join(difflib.unified_diff((BASE_CODE/name).read_text(encoding='utf-8-sig').splitlines(True),s.splitlines(True),fromfile='frozen/'+name,tofile='report_only/'+target.name))
    with open(OUT/(name+'.patch'),'x',encoding='utf-8') as f:f.write(patch)
    if name=='stage1_report.py':
        old=ast.parse((BASE_CODE/name).read_text(encoding='utf-8-sig'));new=ast.parse(s)
        for n in ('welch','rule'):
            a=next(x for x in old.body if isinstance(x,ast.FunctionDef) and x.name==n)
            b=next(x for x in new.body if isinstance(x,ast.FunctionDef) and x.name==n)
            assert ast.dump(a)==ast.dump(b)
v=base.r.read(OUT/'SEED207_VECTOR_AUDIT.json')
v['source_lines']={'preallocation':40,'per_step_assignment':47,'JSONL_flush':51,'JSONL_fsync':67,'normal_completion_vector_flush':69}
# This is a new report-only evidence file, not an original artifact.
(OUT/'SEED207_VECTOR_AUDIT.json').write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
records=dict(base.r.read(OLD/'protected_manifest.json')['sha256'])
for directory in (OLD,BASE_CODE):
    for p in directory.rglob('*'):
        if Path(base.r.winpath(p)).is_file():records[str(p)]=base.r.sha(p)
base.r.write(OUT/'protected_manifest.json',dict(count=len(records),sha256=records,time=time.time()))
frozen={str(p):base.r.sha(p) for p in HERE.glob('*.py')}
for name in ('CORRECTION_FIELD_AUDIT.json','SEED207_VECTOR_AUDIT.json','protected_manifest.json','config.json','authorization.json','stage1/training_plan.json','stage1/reference_receipts.json','stage1/protocol.json'):
    frozen[str(OUT/name)]=base.r.sha(OUT/name)
base.r.write(OUT/'REPORT_CODE_FROZEN.json',dict(sha256=frozen,time=time.time()))
base.r.write(OUT/'REPORT_REPAIR_COMMITMENT.json',dict(scope='CPU report repair and read-only vector audit only',
    original_training_frozen_and_failure_evidence_preserved=True,code_frozen_sha256=base.r.sha(OUT/'REPORT_CODE_FROZEN.json'),
    correction_mapping='original root config.actor_loss -> report correction, verbatim',
    no_training_or_optimization_or_environment_steps=True,no_rule_or_data_changes=True,time=time.time()))
print(json.dumps(dict(prepared=True,protected_files=len(records),frozen_files=len(frozen))))
