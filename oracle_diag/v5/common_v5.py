"""Version 5 paths and immutable evidence, independent of either training stack."""
import os, sys, json, hashlib, time, uuid
from pathlib import Path
sys.dont_write_bytecode = True
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'oracle_diag/results/v5_20261013'
B_CODE = ROOT/'reproductions/otdf_v5'
B_OUT = B_CODE/'results_20261013'
V4_OUT = ROOT/'oracle_diag/results/v4_20261011'
A_METHODS = ('CRITIC-ORACLE','CRITIC-SOURCE','Q-ORACLE')
B_METHODS = ('IQL-TARGET','IQL-TARGET+REG','OTDF-NOREG')
PY_A = Path('D:/ANACONDA/envs/offdynamics/python.exe')
PY_B = ROOT/'reproductions/otdf_train/Scripts/python.exe'

def winpath(p):
    return '\\\\?\\'+str(Path(p).resolve()) if os.name=='nt' else str(Path(p).resolve())

def sha(p):
    h=hashlib.sha256()
    with open(winpath(p),'rb') as f:
        for c in iter(lambda:f.read(1<<20),b''): h.update(c)
    return h.hexdigest()

def read(p):
    with open(winpath(p),encoding='utf-8') as f:return json.load(f)

def write(p,obj,replace=False):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    if p.exists() and not replace:raise RuntimeError('Refuse overwrite: '+str(p))
    tmp=p.with_suffix(p.suffix+'.'+uuid.uuid4().hex+'.tmp')
    with open(winpath(tmp),'w',encoding='utf-8') as f:json.dump(obj,f,ensure_ascii=False,indent=2,allow_nan=False)
    os.replace(winpath(tmp),winpath(p))

def fresh(p):
    p=Path(p)
    if p.exists() and any(p.iterdir()):raise RuntimeError('Refuse nonempty directory: '+str(p))
    p.mkdir(parents=True,exist_ok=True)

def check_approval(kind):
    f=OUT/('gpu_'+kind+'_authorization.json')
    if not f.exists():raise RuntimeError('v5 GPU '+kind+' confirmation is pending, as requested in the v5 specification')
    a=read(f);assert a['authorized'] and a['scope']==kind and a['protocol_sha256']==sha(OUT/'protocol.json')
    ready=read(OUT/'cpu_ready.json');assert ready['passed']
    for p,h in ready['code_sha256'].items():assert sha(p)==h,p
    if kind=='formal':assert read(OUT/'smoke_audited.json')['passed']

def protected_check():
    manifest=read(OUT/'protected_manifest.json')['sha256']
    for p,h in manifest.items():assert sha(p)==h,'Protected file changed: '+p
    return len(manifest)
