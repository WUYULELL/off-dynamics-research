"""New authorization: exactly15 Walker BC jobs using unchanged frozen Hopper training code."""
import sys,os,json,time,math,hashlib,subprocess
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
sys.path.insert(0,str(ROOT/'oracle_diag/v9_hopper_label_headroom/stage3'))
import runtime as r
OLD=r.OUT
OUT=ROOT/'outputs/v9_hopper_label_headroom_20261101/walker_stage3'
def jobs():
    a=[j for j in r.read(OLD/'training_plan.json')['jobs'] if j['task']=='walker2d-morph']
    assert len(a)==15 and all(j['seed'] in range(301,306) for j in a)
    return a
def raw_score_record(path,item,every,last):
    from audit import raw_scores
    if item.get('label')!='R13_R5' or item['seed']!=203:return raw_scores(path,item,every,last)
    # Accepted R13 recovery exception:49 original TB points;100k is sealed raw-episode derived.
    import numpy as np
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    receipt=Path(item['derived_receipt']);assert r.sha(receipt)==item['derived_receipt_sha256']
    derived=r.read(receipt);assert derived['not_an_original_TB_measurement'] and not derived['original_TensorBoard_scalar_available']
    assert r.sha(derived['source'])==derived['source_sha256']
    events=r.read(path/'evaluations.json')['events'];expected=list(range(10000,500001,10000));assert [e['step'] for e in events]==expected
    scalars=EventAccumulator(str(Path(r.shortpath(path))/'tb'),size_guidance={'scalars':0}).Reload().Scalars('evaluation/target_score')
    assert [e.step for e in scalars]==[s for s in expected if s!=100000]
    tb={e.step:e.value for e in scalars};spec=r.task_spec('walker2d-morph');values=[]
    for e in events:
        assert e['xml_sha256']==spec['xml_sha256'] and e['timestep']==.002 and e['env_seeds']==list(range(500020,500030))
        values.append(float(np.mean((np.asarray(e['raw_returns'],dtype=np.float64)-spec['score_min'])/(spec['score_max']-spec['score_min'])*100)))
        if e['step']==100000:
            assert e['raw_returns']==derived['raw_returns'] and abs(values[-1]-derived['normalized_mean'])<1e-8
        else:assert abs(values[-1]-tb[e['step']])<1e-4
    assert all(s in tb for s in [460000,470000,480000,490000,500000])
    return dict(Last5=float(np.mean(values[-5:])),Final=values[-1])
def guard():
    r.check_frozen();assert r.read(OUT/'config.json')['scope']=='Walker15_only_then_stop'
    for p,h in r.read(OUT/'FROZEN.json')['sha256'].items():assert r.sha(p)==h,p
def protect(label):
    m=r.read(OUT/'protected_manifest.json')['sha256'];changed=[p for p,h in m.items() if not Path(r.winpath(p)).exists() or r.sha(p)!=h]
    r.write(OUT/('protected_'+label+'.json'),dict(passed=not changed,count=len(m),changed=changed,time=time.time()));assert not changed,changed[:5]
def child(script,args,name,cpu=True):
    cmd=[str(r.PYTHON),'-B',str(script)]+[str(v) for v in args];en=os.environ.copy()
    en['CUDA_VISIBLE_DEVICES']='-1' if cpu else '0';en['PYTHONDONTWRITEBYTECODE']='1';en.pop('ODRL_PAPER_STYLE_SOURCE_TARGET',None)
    with open(r.winpath(OUT/'logs'/(name+'.stdout.log')),'xb') as so,open(r.winpath(OUT/'logs'/(name+'.stderr.log')),'xb') as se:
        p=subprocess.Popen(cmd,cwd=str(ROOT),env=en,stdout=so,stderr=se,creationflags=subprocess.CREATE_NO_WINDOW)
        r.write(OUT/'active_child.json',dict(pid=p.pid,command=cmd,name=name,cpu=cpu,time=time.time()),replace=True);code=p.wait()
    assert code==0,(name,code,'preserve evidence; no automatic retraining')
