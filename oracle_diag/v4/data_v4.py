"""Array-level alignment checks. No shifted target data for source-only learners."""
from support_v4 import *
from methods_v3 import SourceBackup
import numpy as np

FIELDS = [('state', 'observations'), ('action', 'actions'),
          ('next_state', 'next_observations'), ('reward', 'rewards'), ('not_done', 'terminals')]


def verify_source(src, receipt):
    meta = read(receipt)
    assert src.size == meta['n'] == 999322
    for attr, key in FIELDS:
        arr = getattr(src, attr)
        if key in ('rewards', 'terminals'):
            arr = arr.ravel()
        if key == 'terminals':
            arr = (1. - arr).astype(bool)
        assert array_hash(arr) == meta['source_array_sha256'][key], key
    assert sha(meta['raw_indices_path']) == meta['raw_indices_sha256']
    return meta


def single_source_dataset(src, method, provider=None):
    assert method in SINGLE_SOURCE
    if method == 'ORACLE-ONLY':
        assert provider is not None and provider.arrays is not None
        ns, r, nd = (provider.arrays[k] for k in ('next_state', 'reward', 'not_done'))
    else:
        ns, r, nd = src.next_state, src.reward, src.not_done
    return dict(observations=src.state, actions=src.action,
                next_observations=ns, rewards=r.reshape(-1), terminals=(nd.reshape(-1) == 0))


def verify_single_buffer(src, tar, method, provider=None):
    assert src.size == tar.size == 999322
    assert np.array_equal(src.state, tar.state) and np.array_equal(src.action, tar.action)
    for key in ('next_state', 'reward', 'not_done'):
        expected = provider.arrays[key] if method == 'ORACLE-ONLY' else getattr(src, key)
        assert np.array_equal(getattr(tar, key), expected), key
    return dict(all_rows_equal=True, row_count=src.size, state_action_from_source=True,
                target_dataset_used=False, empirical_target_normalization_used=False,
                target_buffer_slot_contains='source data', labels=method)


def load_source_cpu():
    """Reproduce frozen D4RL retained rows from raw HDF5, without environment rollout."""
    import h5py
    receipt = read(oracle_receipt())
    source = Path.home() / '.d4rl/datasets/walker2d_medium-v2.hdf5'
    assert sha(source) == receipt['source_dataset_sha256']
    with h5py.File(str(source), 'r') as f:
        ids = np.flatnonzero(~f['timeouts'][:-1].astype(bool))
        stored = np.load(receipt['raw_indices_path'])
        assert np.array_equal(ids, stored)
        data = {}
        for key in ('observations', 'actions', 'next_observations', 'rewards', 'terminals'):
            raw = f['observations' if key == 'next_observations' else key][:]
            arr = raw[ids + 1 if key == 'next_observations' else ids]
            data[key] = arr.astype(bool if key == 'terminals' else np.float32)
            assert array_hash(data[key]) == receipt['source_array_sha256'][key], key
    return data
