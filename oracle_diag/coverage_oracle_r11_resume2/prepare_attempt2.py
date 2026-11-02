"""Repair only the pre-training module-path collision; preserve attempt1."""
import json
import hashlib
import runpy
import sys
from pathlib import Path

NEW_CODE=Path(__file__).resolve().parent
ROOT=NEW_CODE.parents[1]
OLD_CODE=ROOT/'oracle_diag/coverage_oracle_r11_resume1'
OLD_OUT=ROOT/'oracle_diag/results/coverage_oracle_20261016_r11_resume1'
NEW_OUT=ROOT/'oracle_diag/results/coverage_oracle_20261016_r11_resume2'
NEW_DOC=ROOT/'outputs/research_idea_protocol_20261017_r11_resume2'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)

def main():
    assert not (OLD_OUT/'runs/tune_T_t10_h1000000_s101').exists(),'Unexpected training evidence; do not auto-rerun'
    assert read(OLD_OUT/'cpu_acceptance.json')['passed']
    NEW_OUT.mkdir(parents=True,exist_ok=False)
    NEW_DOC.mkdir(parents=True,exist_ok=False)
    changes={}
    for source in OLD_CODE.glob('*.py'):
        if source.name=='bootstrap.py':continue
        text=source.read_text(encoding='utf-8')
        if source.name=='runtime.py':
            text=text.replace('coverage_oracle_20261016_r11_resume1','coverage_oracle_20261016_r11_resume2')
            text=text.replace('research_idea_protocol_20261017_r11_resume1','research_idea_protocol_20261017_r11_resume2')
        if source.name=='resume_train.py':
            assert 'from train import evaluate' in text
            text=text.replace('from train import evaluate',"import runpy\nevaluate=runpy.run_path(str(HERE/'train.py'))['evaluate']")
        with open(NEW_CODE/source.name,'x',encoding='utf-8') as f:f.write(text)
        changes[source.name]=dict(original=str(source),before_sha256=sha(source),after_sha256=sha(NEW_CODE/source.name))
    p=read(OLD_OUT/'protocol.json')
    p['version']='R11-resume2'
    p['recommended_defaults']['new_code_path_proposal']=str(NEW_CODE)
    p['recommended_defaults']['new_result_path_proposal']=str(NEW_OUT)
    p['engineering_attempt2']=dict(reason='evaluate imported root train.py due runtime path precedence',
        original_attempt_preserved=str(OLD_OUT),original_updates=0,fix='explicit file-path import of unchanged evaluator')
    write(NEW_OUT/'protocol.json',p)
    auth=read(OLD_OUT/'stage_authorization.json');auth['protocol_sha256']=sha(NEW_OUT/'protocol.json')
    write(NEW_OUT/'stage_authorization.json',auth)
    for name in ('reuse_receipts.json','T_smoke_audited.json','OVERLAP_LOSSES.json'):
        write(NEW_OUT/name,read(OLD_OUT/name))
    sys.path.insert(0,str(NEW_CODE))
    loaded=runpy.run_path(str(NEW_CODE/'resume_train.py'))
    assert Path(loaded['evaluate'].__code__.co_filename).resolve()==NEW_CODE/'train.py'
    import torch
    import runtime as rt
    assert rt.OUT==NEW_OUT and rt.HERE==NEW_CODE
    assert not torch.cuda.is_initialized()
    for n in ('learner.py','train.py','audit.py','selection.py','decision.py','resume_state.py'):
        assert sha(NEW_CODE/n)==sha(OLD_CODE/n)
    base=read(OLD_OUT/'cpu_acceptance.json')
    base.update(engineering_import_delta_passed=True,unchanged_resume_state_hash=sha(NEW_CODE/'resume_state.py'),
        prior_full_CPU_proof=str(OLD_OUT/'cpu_acceptance.json'),prior_full_CPU_proof_sha256=sha(OLD_OUT/'cpu_acceptance.json'),
        wrong_root_train_module_avoided=True,evaluator_exact_path=str(NEW_CODE/'train.py'),code_sha256=rt.code_hashes())
    write(NEW_OUT/'cpu_acceptance.json',base)
    protected=dict(read(OLD_OUT/'protected_manifest.json')['sha256'])
    for root in (OLD_OUT,OLD_CODE):
        for path in root.rglob('*'):
            if path.is_file() and '__pycache__' not in str(path):protected[str(path)]=rt.sha(path)
    write(NEW_OUT/'protected_manifest.json',dict(sha256=protected,count=len(protected)))
    write(NEW_OUT/'engineering_repair.json',dict(changes=changes,source_attempt=str(OLD_OUT),
        failed_before_run_directory_created=True,prior_training_updates=0,no_scientific_change=True,
        import_test_CPU_only=True,full_CPU_restore_tests_reused=True))
    write(NEW_OUT/'code_frozen.json',dict(sha256=rt.code_hashes(),cpu_acceptance_sha256=sha(NEW_OUT/'cpu_acceptance.json')))
    write(NEW_OUT/'active_execution.json',dict(code=str(NEW_CODE),results=str(NEW_OUT),
        queue_script=str(NEW_CODE/'pipeline.py'),queue_status=str(NEW_OUT/'queue_status.json'),
        inactive_attempts=[str(OLD_OUT),str(ROOT/'oracle_diag/results/coverage_oracle_20261016_r11')],
        resume_step=400000,remaining_updates=600000,scope='same R11; no U/G validation/formal/task2/Hopper'))
    (NEW_DOC/'RECOVERY_SCOPE.md').write_text((ROOT/'outputs/research_idea_protocol_20261017_r11_resume1/RECOVERY_SCOPE.md').read_text(encoding='utf-8')+
        '\n工程attempt1在导入阶段退出，尚无运行目录、无训练更新，已保留。attempt2只修正evaluate的明确文件路径导入，CPU确认使用冻结训练文件中的同一函数；未改变恢复算法、模型或科学协议。\n',encoding='utf-8')
    write(NEW_DOC/'PROTOCOL.json',p)
    print('attempt2 explicit evaluator import passed on CPU; unchanged state-restoration proof retained',flush=True)

if __name__=='__main__':main()
