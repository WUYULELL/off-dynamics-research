"""Data loading for the adaptive-source batch.

* Source (D4RL medium): the hdf5 file is read into a dict exactly like the official loaders read files
  (get_keys + [:] / [()] fallback) and passed to d4rl.qlearning_dataset(None, dataset=dict) of the vendored
  minimal d4rl 1.1 (next state = observations[i+1], timeout rows skipped; official OTDF behaviour).
* Target (OTDF training arrays): call_tar_dataset_copy() is the official algo/utils.py call_tar_dataset loop
  (next state = observations[i+1], N-1 rows); only the file path is passed in (see its docstring).
* Dynamics data (j015a): recorded next_observations, as the j008 model worker (raw_transitions()).
"""
import os
from pathlib import Path
import zipfile

from asrc_common import CODE, HOME, array_digest, read_json, rel, sha256_file

KEYS = ('observations', 'actions', 'next_observations', 'rewards', 'terminals')


def get_keys(h5file):
    """Copy of the official algo/utils.py get_keys (and d4rl offline_env.get_keys)."""
    import h5py
    keys = []

    def visitor(name, item):
        if isinstance(item, h5py.Dataset):
            keys.append(name)

    h5file.visititems(visitor)
    return keys


def read_hdf5_dict(path):
    """hdf5 -> dict, as d4rl OfflineEnv.get_dataset and the official call_tar_dataset read files."""
    import h5py
    data_dict = {}
    with h5py.File(str(path), 'r') as dataset_file:
        for k in get_keys(dataset_file):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]
    return data_dict


def import_d4rl():
    """Import the vendored minimal d4rl 1.1 from code/d4rl (never a site-packages copy)."""
    os.environ.setdefault('D4RL_SUPPRESS_IMPORT_ERROR', '1')
    os.environ.setdefault('D4RL_DATASET_DIR', str(CODE/'private_runtime'/'d4rl_datasets'))
    import sys
    if str(CODE) not in sys.path:
        sys.path.insert(0, str(CODE))
    import d4rl
    assert Path(d4rl.__file__).resolve().parent == (CODE/'d4rl').resolve(), 'd4rl not imported from code/d4rl'
    return d4rl


def load_source_official(path, d4rl=None):
    """D4RL medium hdf5 -> dict -> d4rl.qlearning_dataset(None, dataset=dict) (official reading)."""
    import numpy as np
    d4rl = d4rl or import_d4rl()
    raw = read_hdf5_dict(path)
    n = raw['observations'].shape[0]
    # d4rl get_dataset would reshape (N, 1) rewards / terminals to (N,); the v2 medium files are already (N,)
    assert raw['rewards'].shape == (n,) and raw['terminals'].shape == (n,), 'unexpected reward/terminal shape'
    assert 'timeouts' in raw and raw['timeouts'].shape == (n,), 'D4RL v2 file without timeouts'
    ds = d4rl.qlearning_dataset(None, dataset=raw)
    info = dict(raw_rows=int(n), rows=int(len(ds['observations'])), raw_timeouts=int(np.asarray(raw['timeouts']).sum()),
                raw_terminals=int(np.asarray(raw['terminals']).sum()), qlearning_terminals=int(np.asarray(ds['terminals']).sum()))
    return ds, info


def call_tar_dataset_copy(tar_dataset_path):
    """Official OTDF algo/utils.py call_tar_dataset (commit 41e2e62), the reading code and the loop copied verbatim.

    Changes, and only these: (1) the hdf5 path is an argument (official: built from the task name under
    OTDF/datasets/); (2) _max_episode_steps = 1000 replaces gym.make('<robot>-medium-v2')._max_episode_steps
    (= 1000; D4RL's gym environments are not installed on the server; the value only feeds the episode counter,
    which does not change any returned array).
    """
    import numpy as np
    import h5py
    from tqdm import tqdm
    _max_episode_steps = 1000

    data_dict = {}
    with h5py.File(str(tar_dataset_path), 'r') as dataset_file:
        for k in tqdm(get_keys(dataset_file), desc="load datafile"):
            try:  # first try loading as an array
                data_dict[k] = dataset_file[k][:]
            except ValueError as e:  # try loading as a scalar
                data_dict[k] = dataset_file[k][()]

    dataset = data_dict

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    # count how many trajectories are included, ensure that the quantity of trajectories do not exceed number_of_trajectories
    counter = 0
    for i in range(N-1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i+1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        try:
            reward = dataset['rewards'][i].astype(np.float32)[0]
        except:
            reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == _max_episode_steps - 1)

        if done_bool or final_timestep:
            counter +=1
            episode_step = 0

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    return {
        'observations': np.array(obs_),
        'actions': np.array(action_),
        'next_observations': np.array(next_obs_),
        'rewards': np.array(reward_),
        'terminals': np.array(done_),
    }


def features(ds):
    """[s, a, s'] float32 rows, in the order of make_otdf_cost.filter_dataset (state, action, next_state)."""
    import numpy as np
    x = np.hstack([ds['observations'], ds['actions'], ds['next_observations']])
    assert x.dtype == np.float32, x.dtype
    return x


def dataset_digest(ds):
    out = {k: array_digest(ds[k]) for k in KEYS}
    out['features_sas'] = array_digest(features(ds))
    out['rows'] = int(len(ds['observations']))
    return out


def segments_of_raw(path):
    """Trajectory segment id of every raw row (j009/j008 definition): a segment ends at a terminal, a timeout or
    where next_observations[i] != observations[i+1] (|diff| > 1e-5), and at the last row."""
    import numpy as np
    raw = read_hdf5_dict(path)
    obs = np.asarray(raw['observations'], dtype=np.float64)
    n = len(obs)
    term = np.asarray(raw['terminals']).reshape(-1).astype(bool)
    timeouts = np.asarray(raw['timeouts']).reshape(-1).astype(bool) if 'timeouts' in raw else np.zeros(n, bool)
    cont = np.ones(n, bool)
    if 'next_observations' in raw:
        nxt = np.asarray(raw['next_observations'], dtype=np.float64)
        cont[:-1] = np.all(np.abs(nxt[:-1]-obs[1:]) <= 1e-5, axis=1)
    end = term | timeouts | ~cont
    end[-1] = True
    seg = np.concatenate([[0], np.cumsum(end[:-1])]).astype(np.int64)
    return seg, dict(rows=int(n), segments=int(seg.max()+1), terminals=int(term.sum()), timeouts=int(timeouts.sum()),
                     discontinuities=int((~cont[:-1]).sum()))


def raw_transitions(path, sd, ad, drop_timeouts):
    """Recorded transitions with next_observations (j008 model-worker loader): dict of float64 arrays + segment ends."""
    import numpy as np
    raw = read_hdf5_dict(path)
    obs, act, nxt = [np.asarray(raw[k], dtype=np.float64) for k in KEYS[:3]]
    rew = np.asarray(raw['rewards'], dtype=np.float64).reshape(-1)
    term = np.asarray(raw['terminals']).reshape(-1)
    n = len(obs)
    assert obs.shape == (n, sd) and act.shape == (n, ad) and nxt.shape == (n, sd) and len(rew) == n == len(term)
    assert all(np.isfinite(a).all() for a in (obs, act, nxt, rew)) and np.isin(term, [0, 1]).all()
    term = term.astype(bool)
    timeouts = np.asarray(raw['timeouts']).reshape(-1).astype(bool) if 'timeouts' in raw else np.zeros(n, bool)
    cont = np.zeros(n, bool)
    cont[:-1] = np.all(np.abs(nxt[:-1]-obs[1:]) <= 1e-5, axis=1)
    end = term | timeouts | ~cont
    end[-1] = True
    info = dict(rows=int(n), terminal_rows=int(term.sum()), timeout_rows=int(timeouts.sum()), segments=int(end.sum()))
    if drop_timeouts and timeouts.any():
        keep = ~timeouts
        obs, act, nxt, rew, term, end = obs[keep], act[keep], nxt[keep], rew[keep], term[keep], end[keep]
        info['rows_after_dropping_timeouts'] = int(keep.sum())
    return dict(obs=obs, act=act, nxt=nxt, rew=rew, term=term, end=end), info


def segment_bounds(end):
    import numpy as np
    out, start = [], 0
    for i in np.flatnonzero(end):
        out.append((start, int(i)+1))
        start = int(i)+1
    assert start == len(end)
    return out


# ----------------------------------------------------------------------------- server file locations

def _local_files():
    """Local synthetic test only (ASRC_LOCAL_TEST json, never set on the server): replacement data files."""
    p = os.environ.get('ASRC_LOCAL_TEST')
    return read_json(p) if p else None


def source_path(robot, spec):
    lf = _local_files()
    if lf:
        e = lf['source_files'][robot]
        return Path(e['path']), e['sha256']
    s = spec['source'][robot]
    return HOME/s['server_dir']/s['file'], s['sha256']


def target_path(task, spec, preflight_dirs=(), private_data=None):
    """Locate (and for the ODRL archive, extract into private_data) the target file; SHA-256 is checked."""
    name, digest = task['target_file'], task['target_sha256']
    loc = task['target_location']
    lf = _local_files()
    if lf:
        e = lf['target_files'][task['id']]
        p = Path(e['path'])
        assert p.is_file() and sha256_file(p) == e['sha256'], 'local target file'
        return p
    if loc == 'odrl_extra_raw':
        p = HOME/spec['odrl_extra_raw_dir']/name
    elif loc == 'odrl_zip':
        assert private_data is not None, 'odrl_zip targets need a private data directory'
        p = Path(private_data)/name
        if not p.is_file():
            archive = HOME/spec['odrl_archive']['file']
            assert archive.is_file() and sha256_file(archive) == spec['odrl_archive']['sha256'], 'ODRL archive missing or hash mismatch'
            Path(private_data).mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(str(archive)) as z:
                hits = [i for i in z.infolist() if Path(i.filename).name == name and '/raw/' in i.filename]
                assert len(hits) == 1, 'archive entry for %s: %d matches' % (name, len(hits))
                tmp = p.with_name(p.name+'.part')
                tmp.write_bytes(z.read(hits[0]))
                os.replace(str(tmp), str(p))
    elif loc == 'otdf_official_dataset':
        cands = [CODE/'otdf_official'/'datasets'/name]+[Path(d)/'otdf_official'/'datasets'/name for d in preflight_dirs]
        p = next((c for c in cands if c.is_file()), cands[0])
    else:
        raise ValueError(loc)
    assert p.is_file() and not p.is_symlink(), 'target file not available: %s' % rel(p)
    got = sha256_file(p)
    assert got == digest, 'target SHA-256 mismatch for %s' % name
    return p


def preflight_dir(candidates):
    """First preflight job directory (on the server) whose PREFLIGHT.json reports all_passed; None if none."""
    for name in candidates:
        d = HOME/'auto_jobs'/name/'code'
        f = d/'outputs'/'PREFLIGHT.json'
        if f.is_file():
            try:
                if read_json(f).get('all_passed') is True:
                    return d
            except Exception:
                continue
    return None


def load_task_arrays(task, spec, d4rl=None, preflight_dirs=(), private_data=None):
    """Official OTDF training arrays of one task: (source dict, target dict, info)."""
    sp, sd = source_path(task['robot'], spec)
    assert sp.is_file(), 'source data missing: %s' % rel(sp)
    assert sha256_file(sp) == sd, 'source SHA-256 mismatch: %s' % rel(sp)
    tp = target_path(task, spec, preflight_dirs, private_data)
    src, src_info = load_source_official(sp, d4rl)
    tar = call_tar_dataset_copy(tp)
    return src, tar, dict(source_file=rel(sp), source_sha256=sd, target_file=rel(tp), target_sha256=task['target_sha256'],
                          source=src_info, target_rows=int(len(tar['observations'])))
