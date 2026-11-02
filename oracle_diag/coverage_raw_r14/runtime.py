"""R14 raw-label source strength precheck: authorized 40 runs only."""
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import uuid

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT / 'oracle_diag/results/coverage_raw_20261026_r14'
PRIOR = ROOT / 'oracle_diag/results/coverage_oracle_20261021_r13'
DOC = ROOT / 'outputs/research_idea_protocol_20261026_r14'
PYTHON = Path('D:/ANACONDA/envs/offdynamics/python.exe')
for p in (ROOT, ROOT / 'algo'):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
mu = Path.home() / '.mujoco210/mujoco210'
os.environ['MUJOCO_PY_MUJOCO_PATH'] = str(mu)
os.environ['PATH'] = str(mu / 'bin') + os.pathsep + os.environ.get('PATH', '')

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

def protocol():
    return read(OUT / 'protocol.json')

def seed(purpose, run_seed=0, task='walker2d-morph'):
    value = '20261016|%s|%d|%s' % (task, run_seed, purpose)
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:4], 'little')

def task_spec(task):
    return next(t for t in protocol()['tasks'] if t['task'] == task)

def code_hashes():
    paths = list(HERE.glob('*.py')) + [ROOT/'algo/offline_offline'/n for n in ('iql.py','iql_dyn.py','iql_unc.py')]
    return {str(p):sha(p) for p in sorted(paths)}

def check_frozen():
    auth = read(OUT/'stage_authorization.json')
    assert auth['scope'] == 'R14_raw_source_strength_precheck_40_only'
    assert auth['protocol_sha256'] == sha(OUT/'protocol.json')
    assert auth['new_GPU_jobs'] == 40 and auth['estimator_development_authorized'] is False
    for file in ('code_frozen.json','dependency_hashes.json'):
        for path, expected in read(OUT/file)['sha256'].items():
            assert sha(path) == expected, 'Frozen file changed: '+path

def validate_spec(job):
    assert job['phase'] == 'precheck' and job['steps'] == 500000 and job['temp'] == 10
    assert job['arm'] in ('T','M')
    assert job['source_w'] in ((0.,) if job['arm']=='T' else (.25,1.))
    assert job['source_lambda'] == job['source_w']/(1.+job['source_w'])
    assert job['source_labels'] == ('none' if job['arm']=='T' else 'original_dataset')
    if job['task']=='walker2d-morph':
        assert job['arm']=='M' and 201 <= job['seed'] <= 205
    else:
        assert job['task']=='hopper-morph' and 201 <= job['seed'] <= 210
    assert Path(job['out']).resolve().parent == OUT/'runs'

def check_job(job, for_launch=True):
    check_frozen(); validate_spec(job)
    assert job in read(OUT/'training_plan.json')['jobs']
    acceptance = read(OUT/'cpu_acceptance.json')
    assert acceptance['passed'] and acceptance['CUDA_initialized'] is False
    assert sha(OUT/'cpu_acceptance.json') == read(OUT/'CPU_ACCEPTANCE_COMMITMENT.json')['cpu_acceptance_sha256']
    if for_launch:
        assert not (OUT/'STAGE_COMPLETED.json').exists()
        assert not list(OUT.glob('queue_failure_*.json'))

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

def source_data(task, verify=True):
    import h5py
    import numpy as np
    spec = task_spec(task)
    if verify: assert sha(spec['source_dataset']) == spec['source_dataset_sha256']
    with h5py.File(spec['source_dataset'],'r') as file:
        # Keep exactly the R13/D4RL canonical eligible rows; labels come directly from this HDF5.
        ids = np.flatnonzero(~np.asarray(file['timeouts'][:-1],dtype=bool))
        arrays = [np.asarray(file[k],np.float32)[ids] for k in ('observations','actions','next_observations','rewards','terminals')]
    arrays[3] = arrays[3].reshape(-1,1)
    assert np.isin(arrays[4],[0,1]).all()
    arrays[4] = 1.-arrays[4].reshape(-1,1)
    assert len(ids)==spec['source_rows'] and all(np.isfinite(a).all() for a in arrays)
    assert arrays[0].shape==(len(ids),spec['state_dim']) and arrays[1].shape==(len(ids),spec['action_dim'])
    return arrays, ids

def model_config(temp, steps, run_seed, task, source_w):
    spec=task_spec(task)
    return dict(state_dim=spec['state_dim'], action_dim=spec['action_dim'], hidden_sizes=256,
        max_action=1., gamma=.99, tau=.005, update_interval=2, lam=.7, temp=temp,
        critic_lr=.0003, actor_lr=.0003, max_step=steps, seed=run_seed, task=task,
        source_budget=1., source_loss_weight=source_w)

def stream_hash(run_seed, purpose, n, steps, task):
    import numpy as np
    rng=np.random.RandomState(seed(purpose,run_seed,task)); h=hashlib.sha256()
    remaining=steps*64
    while remaining:
        size=min(remaining,262144)
        h.update(rng.randint(n,size=size).astype('<i8').tobytes()); remaining-=size
    return h.hexdigest()

def protect(label):
    records=read(OUT/'protected_manifest.json')['sha256']
    changed=[p for p,h in records.items() if not Path(winpath(p)).exists() or sha(p)!=h]
    write(OUT/('protected_'+label+'.json'),dict(passed=not changed,count=len(records),changed=changed,time=time.time()))
    assert not changed, changed[:10]

