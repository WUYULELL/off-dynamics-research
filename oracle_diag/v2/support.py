"""V2 paths, immutable artifacts and lightweight helpers (no CUDA imports)."""
import os,sys,json,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;OLD=HERE.parent;ROOT=OLD.parent
OUT=OLD/'results/v2_20261006';PREV=OLD/'results/verified_20261003'
ENVS=['walker2d-kinematic-footjnt','hopper-kinematic-legjnt','halfcheetah-kinematic-footjnt']
for p in [ROOT,ROOT/'algo',OLD,HERE]:sys.path.insert(0,str(p))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['D4RL_SUPPRESS_IMPORT_ERROR']='1'
mu=Path.home()/'.mujoco210/mujoco210'
os.environ['MUJOCO_PY_MUJOCO_PATH']=str(mu)
os.environ['PATH']=str(mu/'bin')+os.pathsep+os.environ.get('PATH','')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8'))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''):h.update(block)
    return h.hexdigest()
def write(p,obj,loader='fixed_v2',replace=False):
    p=Path(p)
    if p.exists() and not replace:raise RuntimeError('Refusing overwrite '+str(p))
    p.parent.mkdir(parents=True,exist_ok=True)
    obj=dict(loader=loader,**obj) if 'loader' not in obj else obj
    tmp=p.with_suffix('.tmp');tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');os.replace(str(tmp),str(p))
def fresh(p):
    if p.exists() and any(p.iterdir()):raise RuntimeError('Refusing overwrite '+str(p))
    p.mkdir(parents=True,exist_ok=True)
def stats(x):
    import numpy as np
    x=np.asarray(x);return dict(n=int(x.size),mean=float(x.mean()),median=float(np.median(x)),p95=float(np.percentile(x,95)),max=float(x.max()))
