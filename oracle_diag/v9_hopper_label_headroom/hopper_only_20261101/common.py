"""User correction: preserve all original science; whitelist Hopper and stop before Walker."""
import sys,os,json,time,hashlib,math,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'oracle_diag/v9_hopper_label_headroom/stage3'))
import runtime as r
OUT=ROOT/'outputs/v9_hopper_label_headroom_20261101/hopper_stage3_correction'
OLD=r.OUT
def guard():
    r.check_frozen()
    assert r.read(OUT/'config.json')['scope']=='Hopper30_only_then_stop_before_Walker'
    for p,h in r.read(OUT/'FROZEN.json')['sha256'].items():assert r.sha(p)==h,p
    assert not list((OLD/'runs').glob('walk_*')),'Walker must not run without new authorization'
    assert not list(OLD.glob('queue_failure_*.json'))
def jobs():
    result=[j for j in r.read(OLD/'training_plan.json')['jobs'] if j['task']=='hopper-morph']
    assert len(result)==30 and all(j['seed'] in range(301,311) for j in result)
    return result
def child(script,args,name,cpu=True):
    cmd=[str(r.PYTHON),'-B',str(script)]+[str(a) for a in args]
    en=os.environ.copy();en['CUDA_VISIBLE_DEVICES']='-1' if cpu else '0';en['PYTHONDONTWRITEBYTECODE']='1';en.pop('ODRL_PAPER_STYLE_SOURCE_TARGET',None)
    with open(r.winpath(OUT/'logs'/(name+'.stdout.log')),'xb') as so,open(r.winpath(OUT/'logs'/(name+'.stderr.log')),'xb') as se:
        p=subprocess.Popen(cmd,cwd=str(ROOT),env=en,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        r.write(OUT/'active_child.json',dict(pid=p.pid,command=cmd,name=name,cpu=cpu,time=time.time()),replace=True)
        rc=p.wait()
    assert rc==0,(name,rc,'preserve evidence; no automatic retraining')
