"""V3-only paths, immutable writes and runtime setup. No old artifacts are modified."""
import os, sys, json, hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
OLD=HERE.parent
ROOT=OLD.parent
OUT=OLD/'results/v3_20261008'
V2=OLD/'results/v2_20261006'
for p in [ROOT,ROOT/'algo',HERE]:
    if str(p) not in sys.path: sys.path.insert(0,str(p))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['D4RL_SUPPRESS_IMPORT_ERROR']='1'
mu=Path.home()/'.mujoco210/mujoco210'
os.environ['MUJOCO_PY_MUJOCO_PATH']=str(mu)
os.environ['PATH']=str(mu/'bin')+os.pathsep+os.environ.get('PATH','')

def winpath(path):
    return '\\\\?\\'+str(Path(path).resolve()) if os.name=='nt' else str(Path(path).resolve())

def read(path):
    with open(winpath(path),'r',encoding='utf-8') as f: return json.load(f)

def sha(path):
    h=hashlib.sha256()
    with open(winpath(path),'rb') as f:
        for chunk in iter(lambda:f.read(1<<20),b''):h.update(chunk)
    return h.hexdigest()

def write(path,value,replace=False):
    path=Path(path)
    if path.exists() and not replace:raise RuntimeError('Refusing overwrite '+str(path))
    path.parent.mkdir(parents=True,exist_ok=True)
    value=dict(loader='explicit_next_v3',**value) if 'loader' not in value else value
    temporary=path.with_suffix('.tmp')
    with open(winpath(temporary),'w',encoding='utf-8') as f:json.dump(value,f,indent=2,ensure_ascii=False,allow_nan=False)
    os.replace(winpath(temporary),winpath(path))

def fresh(path):
    path=Path(path)
    if path.exists() and any(path.iterdir()):raise RuntimeError('Refusing nonempty directory '+str(path))
    path.mkdir(parents=True,exist_ok=True)

def stats(values):
    import numpy as np
    values=np.asarray(values)
    assert values.size and np.isfinite(values).all()
    return dict(n=int(values.size),mean=float(values.mean()),median=float(np.median(values)),
                p95=float(np.percentile(values,95)),max=float(values.max()))

def protected_check():
    for path,expected in read(OUT/'protected_manifest.json')['sha256'].items():
        assert sha(path)==expected,'Protected artifact changed: '+path
