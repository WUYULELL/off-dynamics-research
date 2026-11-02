"""V9 stage1-only CPU paths, immutable artifacts and simulator adapters."""
import os
import sys
import time
import json
import hashlib
import uuid
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1'
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['D4RL_SUPPRESS_IMPORT_ERROR']='1'
os.environ['OMP_NUM_THREADS']='2'
os.environ['MKL_NUM_THREADS']='2'
os.environ.pop('ODRL_PAPER_STYLE_SOURCE_TARGET',None)
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
BASE=ROOT/'outputs/v9_hopper_label_headroom_20261030'
OUT=BASE/'stage1'
PYTHON=Path('D:/ANACONDA/envs/offdynamics/python.exe')
SOURCE=Path.home()/'.d4rl/datasets/hopper_medium-v2.hdf5'
XML=ROOT/'reproductions/OTDF/envs/assets/hopper_morph.xml'
mu=Path.home()/'.mujoco210/mujoco210'
os.environ['MUJOCO_PY_MUJOCO_PATH']=str(mu)
os.environ['PATH']=str(mu/'bin')+os.pathsep+os.environ.get('PATH','')

def wp(p):
    p=str(Path(p).resolve())
    return '\\\\?\\'+p if os.name=='nt' and not p.startswith('\\\\?\\') else p

def sha(p):
    h=hashlib.sha256()
    with open(wp(p),'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def read(p):
    with open(wp(p),'r',encoding='utf-8-sig') as f:return json.load(f)

def write(p,value,replace=False):
    p=Path(p)
    if p.exists() and not replace:raise FileExistsError(str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    tmp=p.with_name(p.name+'.'+uuid.uuid4().hex+'.tmp')
    with open(wp(tmp),'w',encoding='utf-8') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.flush();os.fsync(f.fileno())
    for attempt in range(8):
        try:os.replace(wp(tmp),wp(p));return
        except PermissionError:
            if attempt==7:raise
            time.sleep(min(.2*(attempt+1),1.))

def stats(x):
    import numpy as np
    x=np.asarray(x,dtype=np.float64)
    assert len(x) and np.isfinite(x).all()
    return dict(n=int(x.size),mean=float(x.mean()),median=float(np.median(x)),p99=float(np.percentile(x,99)),maximum=float(x.max()))

def check_frozen():
    assert read(OUT/'authorization.json')['stage']==1
    assert not read(OUT/'authorization.json')['RL_training_authorized']
    for n in ('code_frozen.json','dependency_hashes.json'):
        for p,h in read(OUT/n)['sha256'].items():assert sha(p)==h,'Frozen input changed: '+p

def protected_check(label):
    d=read(OUT/'protected_manifest.json')['sha256']
    changed=[p for p,h in d.items() if not Path(wp(p)).exists() or sha(p)!=h]
    write(OUT/('protected_'+label+'.json'),dict(passed=not changed,count=len(d),changed=changed,time=time.time()))
    assert not changed,changed[:10]

def load_raw():
    import numpy as np
    import h5py
    with h5py.File(wp(SOURCE),'r') as f:
        r={k:np.asarray(f[k]) for k in ('observations','actions','next_observations','rewards','terminals','timeouts','infos/qpos','infos/qvel')}
    assert len(r['rewards'])==1000000 and r['observations'].shape==(1000000,11)
    return r

def make_envs():
    import gym
    import d4rl
    from gym.envs.mujoco.hopper_v3 import HopperEnv
    source=gym.make('hopper-medium-v2')
    target=gym.wrappers.TimeLimit(HopperEnv(xml_file=str(XML)),max_episode_steps=1000)
    source.seed(20261030);target.seed(20261031)
    assert source.unwrapped.model.nq==target.unwrapped.model.nq==6
    assert source.unwrapped.model.nv==target.unwrapped.model.nv==6
    assert source.unwrapped.frame_skip==target.unwrapped.frame_skip==4
    assert abs(float(target.unwrapped.model.opt.timestep)-.002)<1e-12
    g=target.unwrapped.model.geom_name2id('torso_geom')
    assert abs(float(target.unwrapped.model.geom_size[g,0])-.125)<1e-12
    return source,target

def cold_step(env,p,v,a):
    import numpy as np
    u=env.unwrapped;u.sim.reset()
    u.set_state(np.asarray(p,np.float64),np.asarray(v,np.float64))
    u.sim.data.qacc_warmstart[:]=0
    restored=np.asarray(u._get_obs()).copy()
    ns,r,d,_=u.step(np.asarray(a,np.float64))
    return np.asarray(ns).copy(),float(r),bool(d),u.sim.data.qpos.copy(),u.sim.data.qvel.copy(),restored
