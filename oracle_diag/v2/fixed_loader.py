"""Explicit next observations, all N rows; original reward/done/dtype semantics."""
from support import ROOT
import h5py,numpy as np
def source_path(env_name,shift_level='medium',quality='medium'):
    return ROOT/'dataset/mujoco'/('{}_{}_{}.hdf5'.format(env_name.replace('-','_'),shift_level,quality))
def load_file(path):
    with h5py.File(path,'r') as f:
        data={k:f[k][:] for k in ['observations','actions','next_observations','rewards','terminals']}
    assert len({len(x) for x in data.values()})==1
    return dict(observations=data['observations'].astype(np.float32),actions=data['actions'].astype(np.float32),
                next_observations=data['next_observations'].astype(np.float32),
                rewards=data['rewards'].astype(np.float32).reshape(-1),terminals=data['terminals'].astype(bool).reshape(-1))
def load_source_fixed(env_name,shift_level,quality='random'):
    return load_file(source_path(env_name,shift_level,quality))
