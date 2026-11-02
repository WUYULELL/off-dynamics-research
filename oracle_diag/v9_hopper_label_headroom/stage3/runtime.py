import os,sys,json,time,uuid,hashlib,math
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[2]
OUT=ROOT/'outputs/v9_hopper_label_headroom_20261031/stage3'
STAGE2=ROOT/'outputs/v9_hopper_label_headroom_20261030/stage2'
PYTHON=Path('D:/ANACONDA/envs/offdynamics/python.exe')
for p in (ROOT,ROOT/'algo'):
    if str(p) not in sys.path:sys.path.insert(0,str(p))
os.environ['PYTHONDONTWRITEBYTECODE']='1'
os.environ['D4RL_SUPPRESS_IMPORT_ERROR']='1'
os.environ.pop('ODRL_PAPER_STYLE_SOURCE_TARGET',None)
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
mu=Path.home()/'.mujoco210/mujoco210'
os.environ['MUJOCO_PY_MUJOCO_PATH']=str(mu)
os.environ['PATH']=str(mu/'bin')+os.pathsep+os.environ.get('PATH','')
def winpath(path):
    s = str(Path(path).resolve())
    return s if os.name != 'nt' or s.startswith('\\\\?\\') else '\\\\?\\' + s

def shortpath(path):
    if os.name != 'nt':
        return str(path)
    import ctypes
    buf = ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(Path(path).resolve()), buf, len(buf)):
        raise RuntimeError('Cannot obtain short Windows path: ' + str(path))
    return buf.value

def sha(path):
    h = hashlib.sha256()
    with open(winpath(path), 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()

def read(path):
    return json.loads(Path(winpath(path)).read_text(encoding='utf-8-sig'))

def write(path, value, replace=False):
    path = Path(path)
    if path.exists() and not replace:
        raise FileExistsError(str(path))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with open(tmp, 'w', encoding='utf-8') as f:
        json.dump(value, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    for i in range(8):
        try:
            os.replace(str(tmp), str(path))
            return
        except PermissionError:
            if i == 7:
                raise
            time.sleep(min(.2 * (i + 1), 1.))

def fresh(path):
    Path(path).mkdir(parents=True, exist_ok=False)

def configure_cpu():
    import torch
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)

def env(task):
    import gym
    from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
    from gym.envs.mujoco.hopper_v3 import HopperEnv
    spec = task_spec(task)
    assert sha(spec['xml']) == spec['xml_sha256']
    constructor = Walker2dEnv if task=='walker2d-morph' else HopperEnv
    return gym.wrappers.TimeLimit(constructor(xml_file=spec['xml']), max_episode_steps=1000)

def target_data(task):
    import h5py
    import numpy as np
    spec = task_spec(task)
    assert sha(spec['target_dataset']) == spec['target_dataset_sha256']
    with h5py.File(spec['target_dataset'],'r') as file:
        arrays = [np.asarray(file[k],np.float32) for k in ('observations','actions','next_observations','rewards','terminals')]
    arrays[3] = arrays[3].reshape(-1,1)
    assert np.isin(arrays[4],[0,1]).all()
    arrays[4] = 1.-arrays[4].reshape(-1,1)
    assert all(len(a)==5000 and np.isfinite(a).all() for a in arrays)
    assert arrays[0].shape==(5000,spec['state_dim']) and arrays[1].shape==(5000,spec['action_dim'])
    return arrays

def protocol():return read(OUT/'protocol.json')
def task_spec(task):return next(v for v in protocol()['tasks'] if v['task']==task)
def seed(purpose,run_seed,task):
    return int.from_bytes(hashlib.sha256(('V9-stage3-20261031|%s|%d|%s'%(task,run_seed,purpose)).encode()).digest()[:4],'little')
def code_hashes():
    return {str(p):sha(p) for p in sorted(list(HERE.glob('*.py'))+[ROOT/'algo/offline_offline/iql.py',ROOT/'algo/offline_offline/iql_dyn.py',ROOT/'algo/offline_offline/iql_unc.py'])}
def check_frozen():
    a=read(OUT/'authorization.json');assert a['new_jobs']==45 and a['order']==['hopper-morph','walker2d-morph']
    for p,h in read(OUT/'code_frozen.json')['sha256'].items():assert sha(p)==h,p
    for p,h in read(OUT/'input_frozen.json')['sha256'].items():assert sha(p)==h,p
    assert read(OUT/'cpu_acceptance.json')['passed']
    assert sha(OUT/'cpu_acceptance.json')==read(OUT/'CPU_ACCEPTANCE_COMMITMENT.json')['sha256']
def ids(job):return read(OUT/'selections.json')[job['task']]['variants'][job['variant']]['rows']
def sample_hash(job,steps=200000):
    import numpy as np
    rows=np.asarray(ids(job),dtype=np.int64);rng=np.random.RandomState(seed('target_sample',job['seed'],job['task']));h=hashlib.sha256()
    remaining=steps*64
    while remaining:
        n=min(remaining,262144);h.update(rows[rng.randint(len(rows),size=n)].astype('<i8').tobytes());remaining-=n
    return h.hexdigest()
def validate(job):
    check_frozen();assert job in read(OUT/'training_plan.json')['jobs']
    assert job['steps']==200000 and job['beta']==0 and job['batch']==64
def protect(label):
    m=read(OUT/'protected_manifest.json')['sha256'];changed=[p for p,h in m.items() if not Path(winpath(p)).exists() or sha(p)!=h]
    write(OUT/('protected_'+label+'.json'),dict(passed=not changed,count=len(m),changed=changed,time=time.time()))
    assert not changed,changed[:10]
