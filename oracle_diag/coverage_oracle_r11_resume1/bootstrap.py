"""New recovery directory, old science and outputs stay immutable."""
import hashlib
import json
from pathlib import Path

NEW_CODE=Path(__file__).resolve().parent
ROOT=NEW_CODE.parents[1]
OLD_CODE=ROOT/'oracle_diag/coverage_oracle_r11'
OLD_OUT=ROOT/'oracle_diag/results/coverage_oracle_20261016_r11'
NEW_OUT=ROOT/'oracle_diag/results/coverage_oracle_20261016_r11_resume1'
NEW_DOC=ROOT/'outputs/research_idea_protocol_20261017_r11_resume1'

def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2,allow_nan=False)

def main():
    NEW_OUT.mkdir(parents=True,exist_ok=False)
    NEW_DOC.mkdir(parents=True,exist_ok=False)
    copies={}
    for n in ('learner.py','train.py','audit.py','selection.py','decision.py'):
        target=NEW_CODE/n
        with open(target,'xb') as f:f.write((OLD_CODE/n).read_bytes())
        copies[n]=dict(source=str(OLD_CODE/n),sha256=sha(target))
        assert sha(target)==sha(OLD_CODE/n)
    text=(OLD_CODE/'runtime.py').read_text(encoding='utf-8')
    text=text.replace('coverage_oracle_20261016_r11','coverage_oracle_20261016_r11_resume1')
    text=text.replace('research_idea_protocol_20261016_r11','research_idea_protocol_20261017_r11_resume1')
    with open(NEW_CODE/'runtime.py','x',encoding='utf-8') as f:f.write(text)
    text=(OLD_CODE/'pipeline.py').read_text(encoding='utf-8')
    text=text.replace('reused_jobs=2','reused_jobs=10')
    text=text.replace('logical_T_jobs=12,reused_T_jobs=1,new_T_jobs=11','logical_T_jobs=12,reused_T_jobs=9,new_T_jobs=3')
    text=text.replace('new_GPU_job_total=17','new_GPU_job_total=9')
    text=text.replace("self.child('train.py',['--job',file],ident)",
        "runner='resume_train.py' if phase=='tune' and h==1000000 and s==101 else 'train.py'\n        self.child(runner,['--job',file],ident)")
    a=text.index('    def T500(')
    b=text.index('    def main(',a)
    text=text[:a]+'''    def T500(self,t,s):
        receipt=self.reuse['T500_by_temp_seed']['%g_%d'%(t,s)]
        assert sha(receipt['audit'])==receipt['audit_sha256']
        return read(receipt['audit'])

'''+text[b:]
    text=text.replace('本轮新增12次长训练＋5次5k冒烟；T冒烟另复用旧合格结果，没有重跑。',
        '本次恢复新增4个长训练作业（含400k续至1M）＋5次5k冒烟；9项500k和T冒烟复用，未重跑。')
    text=text.replace('reused_jobs=2,new_GPU_jobs=17','reused_jobs=10,new_GPU_jobs=9')
    # This report retains the original R11 filenames and scientific decision rules.
    with open(NEW_CODE/'pipeline.py','x',encoding='utf-8') as f:f.write(text)
    p=read(OLD_OUT/'protocol.json')
    p['version']='R11-resume1'
    p['recovery']=dict(user_request='可以从中断的位置继续跑吗',
        checkpoint_step=400000,interrupted_record_step=407000,global_budget=1000000,
        remaining_updates=600000,recomputed_discarded_updates=7000,
        checkpoint=str(OLD_OUT/'runs/tune_T_t10_h1000000_s101/model_400000.pt'),
        checkpoint_sha256='0951586da970e5c7f17a0759715801252df6657175d735d25cddc6f68748adf9',
        original_horizon_and_schedule_preserved=True,restore_weights_target_Q_optimizers_scheduler_and_counter=True,
        sampler='rebuild deterministic stream and match stored 407k/500k hashes',
        overlap_gate='every available 401k-406k GPU loss must match original float32 exactly; otherwise stop',
        old_outputs_read_only=True,new_attempt_only=True)
    p['stage_scope']['already_complete_long']=9
    p['stage_scope']['new_long_runs']=4
    p['stage_scope']['new_smoke_runs']=5
    p['recommended_defaults']['new_code_path_proposal']=str(NEW_CODE)
    p['recommended_defaults']['new_result_path_proposal']=str(NEW_OUT)
    write(NEW_OUT/'protocol.json',p)
    auth=read(OLD_OUT/'stage_authorization.json')
    auth['protocol_sha256']=sha(NEW_OUT/'protocol.json')
    auth['recovery_request']='用户询问从中断位置继续；只恢复400k检查点至原1M，继承R11分段边界，不从零重跑成功作业'
    write(NEW_OUT/'stage_authorization.json',auth)
    write(NEW_OUT/'copy_provenance.json',dict(unchanged_science_files=copies,
        old_runtime_sha256=sha(OLD_CODE/'runtime.py'),old_pipeline_sha256=sha(OLD_CODE/'pipeline.py'),
        changed='isolated paths, all9 T500k reuse, one full-state continuation runner; no science changes'))
    write(NEW_OUT/'active_execution.json',dict(code=str(NEW_CODE),results=str(NEW_OUT),
        queue_script=str(NEW_CODE/'pipeline.py'),queue_status=str(NEW_OUT/'queue_status.json'),
        replaces_inactive_R11=str(OLD_OUT),resume_step=400000,remaining_updates=600000,
        scope='same R11: no U/G validations, no formal/task2/Hopper'))
    (NEW_DOC/'RECOVERY_SCOPE.md').write_text('''# R11-resume1 恢复范围

最近可恢复点为400k，407k只有进度记录。完整恢复policy/V/Q/targetQ、三个Adam状态、原1M周期cosine调度及计数；抽样器按原seed重建到400k，并与旧407k和完成500k抽样哈希独立核验。

从400k运行至原1M，总预算不变，剩余600k；400k–407k之间7000步重新计算，原尝试保留。原有9项500k及T冒烟复用，不重跑。新attempt的前缀日志只继承到400k，丢弃旧400k之后数据，避免重复或混用。

CPU连续/恢复一致性和状态验收先通过；新GPU训练在401k–406k已有记录上逐项核对V/Q/actor的float32损失，严格相等才继续。若不匹配，不放宽容差、不进入新步数，停止报告。

仍仅R11已授权T12、参考1、5000步CPU诊断、零新增训练分布审计、五组5k冒烟。U3/G9验证、C网格、正式、第二任务/Hopper缓存均未授权。温度锁定10、验证seed101–103和H判据不改；两档Last-5窗口不同阶段的限定不改。
''',encoding='utf-8')
    print(str(NEW_OUT),flush=True)

if __name__=='__main__':main()
