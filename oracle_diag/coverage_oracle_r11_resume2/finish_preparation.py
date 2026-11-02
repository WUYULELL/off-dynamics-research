"""Complete unfrozen attempt2 preparation after a text newline byte-check failure."""
import hashlib,json,runpy,sys
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
    with open(p,'x',encoding='utf-8') as f:json.dump(x,f,indent=2,ensure_ascii=False,allow_nan=False)

def main():
    assert not (NEW_OUT/'code_frozen.json').exists() and not (NEW_OUT/'queue_started.json').exists()
    changes=[]
    names=('learner.py','train.py','audit.py','selection.py','decision.py','resume_state.py')
    for n in names:
        a,b=NEW_CODE/n,OLD_CODE/n
        assert a.read_text(encoding='utf-8')==b.read_text(encoding='utf-8')
        before=sha(a)
        a.write_bytes(b.read_bytes())
        assert sha(a)==sha(b)
        changes.append(dict(file=n,before_sha256=before,after_sha256=sha(a),text_identical=True))
    write(NEW_OUT/'preparation_newline_check.json',dict(failure='Windows text writer translated new copied files to CRLF; byte-check caught it before freeze or training',
        repaired='restore exact original bytes for unchanged science modules',changes=changes,training_updates=0))
    sys.path.insert(0,str(NEW_CODE))
    loaded=runpy.run_path(str(NEW_CODE/'resume_train.py'))
    assert Path(loaded['evaluate'].__code__.co_filename).resolve()==NEW_CODE/'train.py'
    import torch
    import runtime as rt
    assert rt.OUT==NEW_OUT and rt.HERE==NEW_CODE and not torch.cuda.is_initialized()
    base=read(OLD_OUT/'cpu_acceptance.json')
    base.update(engineering_import_delta_passed=True,unchanged_resume_state_hash=sha(NEW_CODE/'resume_state.py'),
        prior_full_CPU_proof=str(OLD_OUT/'cpu_acceptance.json'),prior_full_CPU_proof_sha256=sha(OLD_OUT/'cpu_acceptance.json'),
        wrong_root_train_module_avoided=True,evaluator_exact_path=str(NEW_CODE/'train.py'),code_sha256=rt.code_hashes())
    write(NEW_OUT/'cpu_acceptance.json',base)
    protected=dict(read(OLD_OUT/'protected_manifest.json')['sha256'])
    for root in (OLD_OUT,OLD_CODE):
        for p in root.rglob('*'):
            if p.is_file() and '__pycache__' not in str(p):protected[str(p)]=rt.sha(p)
    write(NEW_OUT/'protected_manifest.json',dict(sha256=protected,count=len(protected)))
    write(NEW_OUT/'engineering_repair.json',dict(before={str(p):sha(p) for p in OLD_CODE.glob('*.py')},
        after={str(p):sha(p) for p in NEW_CODE.glob('*.py')},original_attempt=str(OLD_OUT),
        prior_training_updates=0,no_scientific_change=True,
        fix='explicit file-path evaluator import; original science modules copied byte-for-byte'))
    write(NEW_OUT/'code_frozen.json',dict(sha256=rt.code_hashes(),cpu_acceptance_sha256=sha(NEW_OUT/'cpu_acceptance.json')))
    write(NEW_OUT/'active_execution.json',dict(code=str(NEW_CODE),results=str(NEW_OUT),
        queue_script=str(NEW_CODE/'pipeline.py'),queue_status=str(NEW_OUT/'queue_status.json'),
        inactive_attempts=[str(OLD_OUT),str(ROOT/'oracle_diag/results/coverage_oracle_20261016_r11')],
        resume_step=400000,remaining_updates=600000,scope='same R11; no U/G validation/formal/task2/Hopper'))
    (NEW_DOC/'RECOVERY_SCOPE.md').write_text((ROOT/'outputs/research_idea_protocol_20261017_r11_resume1/RECOVERY_SCOPE.md').read_text(encoding='utf-8')+
        '\n工程attempt1在导入阶段退出，无运行目录、无训练更新，已保留。attempt2仅修正evaluate的明确路径导入，并通过CPU入口测试；未修改恢复算法/模型/科学协议。准备阶段字节验收检出Windows换行转换，已在冻结前恢复逐字节原副本，训练更新为0。\n',encoding='utf-8')
    write(NEW_DOC/'PROTOCOL.json',read(NEW_OUT/'protocol.json'))
    print('attempt2 CPU import/byte checks passed; full state and stream proof unchanged',flush=True)

if __name__=='__main__':main()
