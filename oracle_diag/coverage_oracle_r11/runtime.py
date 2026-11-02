"""Isolated R11 runtime. Old experiments are read-only dependencies."""
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
OUT = ROOT / 'oracle_diag/results/coverage_oracle_20261016_r11'
DOC = ROOT / 'outputs/research_idea_protocol_20261016_r11'
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

def seed(purpose, run_seed=0):
    s = '20261016|walker2d-morph|%d|%s' % (run_seed, purpose)
    return int.from_bytes(hashlib.sha256(s.encode()).digest()[:4], 'little')

def code_hashes():
    files = list(HERE.glob('*.py')) + [ROOT / 'algo/offline_offline/iql_dyn.py',
        ROOT / 'algo/offline_offline/iql_unc.py', ROOT / 'algo/offline_offline/iql.py']
    return {str(p): sha(p) for p in sorted(files)}

def check_frozen():
    a=read(OUT/'stage_authorization.json')
    assert a['scope']=='T12_reference_diagnosis_scores_five_smokes_only'
    assert a['protocol_sha256']==sha(OUT/'protocol.json')
    assert all(a[k] is False for k in ('U_validation_authorized','G_validation_authorized',
        'C_grid_authorized','formal_authorized','task2_authorized','Hopper_cache_authorized'))
    for path,expected in read(OUT/'code_frozen.json')['sha256'].items():
        assert sha(path)==expected,'Code changed after freeze: '+path

def check_job(job,for_launch=True):
    check_frozen()
    assert job['task']=='walker2d-morph'
    arm,phase,s,h=(job[k] for k in ('arm','phase','seed','steps'))
    assert job['temp'] in (1.,3.,10.)
    if phase=='tune':
        assert arm=='T' and s in (101,102,103) and h in (500000,1000000)
        if h==1000000:
            chosen=read(OUT/'TEMP_500K_SELECTED.json')
            assert job['temp']==chosen['temp'],'1M temperature must stay locked to 500k winner'
    elif phase=='reference':
        assert arm=='T' and s==1001
        chosen=read(OUT/'T_selected.json')
        assert job['temp']==chosen['temp'] and h==chosen['steps']
    elif phase=='smoke':
        assert arm in ('G','C','R','U','D') and s==7001 and h==5000
        chosen=read(OUT/'T_selected.json')
        assert job['temp']==chosen['temp']
        assert read(OUT/'scores/completed.json')['passed']
        if arm in ('G','C','R'):assert job['ratio']==.2
    else:
        raise AssertionError('Not authorized in R11: '+phase)
    assert read(OUT/'cpu_acceptance.json')['passed']
    assert read(OUT/'T_smoke_audited.json')['passed']
    assert OUT in Path(job['out']).resolve().parents
    if for_launch:
        assert not (OUT/'STAGE_COMPLETED.json').exists(),'Stage ended; separate U/G authorization required'

def configure_cpu():
    import torch
    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    torch.use_deterministic_algorithms(True)

def env():
    import gym
    from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
    task = protocol()['tasks'][0]
    assert sha(task['xml']) == task['xml_sha256']
    return gym.wrappers.TimeLimit(Walker2dEnv(xml_file=task['xml']), max_episode_steps=1000)

def target_data():
    import h5py
    import numpy as np
    task = protocol()['tasks'][0]
    assert sha(task['target_dataset']) == task['target_dataset_sha256']
    with h5py.File(task['target_dataset'], 'r') as f:
        a = [np.asarray(f[k], dtype=np.float32) for k in
             ('observations', 'actions', 'next_observations', 'rewards', 'terminals')]
    a[3], a[4] = a[3].reshape(-1, 1), 1. - a[4].reshape(-1, 1)
    assert all(len(x) == 5000 and np.isfinite(x).all() for x in a)
    assert a[0].shape == (5000, 17) and a[1].shape == (5000, 6)
    return a

def source_data(verify=True):
    import h5py
    import numpy as np
    task = protocol()['tasks'][0]
    receipt = ROOT / 'oracle_diag/results/v3_20261008/source_oracle1/completed.json'
    meta = read(receipt)
    assert meta['complete'] and not meta['failed_raw_rows'] and meta['n'] == 999322
    assert meta['source_dataset_sha256'] == task['source_dataset_sha256']
    assert meta['xml_sha256'] == task['xml_sha256']
    if verify:
        assert sha(task['source_dataset']) == task['source_dataset_sha256']
    with h5py.File(task['source_dataset'], 'r') as f:
        keep = np.flatnonzero(~np.asarray(f['timeouts'][:-1], dtype=bool))
        obs = np.asarray(f['observations'], dtype=np.float32)
        raw = [obs[keep], np.asarray(f['actions'], dtype=np.float32)[keep], obs[keep + 1],
               np.asarray(f['rewards'], dtype=np.float32)[keep].reshape(-1, 1),
               1. - np.asarray(f['terminals'], dtype=np.float32)[keep].reshape(-1, 1)]
    assert len(keep) == meta['n']
    if verify:
        assert sha(meta['raw_indices_path']) == meta['raw_indices_sha256']
    assert np.array_equal(keep, np.load(meta['raw_indices_path']))
    oracle = []
    for key in ('next_state', 'reward', 'not_done'):
        rec = meta['arrays'][key]
        if verify:
            assert sha(rec['path']) == rec['sha256']
        v = np.load(rec['path'], mmap_mode='r')
        assert len(v) == len(keep) and np.isfinite(v).all()
        oracle.append(v.reshape((len(keep), -1)))
    return raw, [raw[0], raw[1]] + oracle, keep, meta

def model_config(temp, steps, run_seed):
    return dict(state_dim=17, action_dim=6, hidden_sizes=256, max_action=1.,
        gamma=.99, tau=.005, update_interval=2, lam=.7, temp=temp,
        critic_lr=.0003, actor_lr=.0003, max_step=steps, seed=run_seed,
        source_budget=1., source_loss_weight=1.)
