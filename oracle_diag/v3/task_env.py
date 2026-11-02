"""Strict OTDF environment/dataset adapters, patched only into the v3 process."""
from support_v3 import ROOT,sha
import runpy
import numpy as np
import h5py

TASKS={
    'walker2d-morph':('walker_morph.xml','walker2d_morph_medium.hdf5'),
    'walker2d-gravity':('walker_gravity.xml','walker2d_gravity_0.5_medium.hdf5'),
    'hopper-morph':('hopper_morph.xml','hopper_morph_medium.hdf5'),
    'hopper-gravity':('hopper_gravity.xml','hopper_gravity_0.5_medium.hdf5'),
}

def paths(task):
    if task not in TASKS:raise ValueError('Unsupported v3 task: '+str(task))
    xml,data=TASKS[task]
    return ROOT/'reproductions/OTDF/envs/assets'/xml,ROOT/'reproductions/OTDF/datasets'/data

def normalization(task):
    paths(task)
    file=ROOT/'reproductions/OTDF/info.py'
    constants=runpy.run_path(str(file))
    return dict(task=task,reference='OTDF/info.py',file=str(file),sha256=sha(file),
                min=float(constants['REF_MIN_SCORE'][task]),max=float(constants['REF_MAX_SCORE'][task]))

def call_shifted_env(env_config):
    import gym
    from gym.envs.mujoco.walker2d_v3 import Walker2dEnv
    from gym.envs.mujoco.hopper_v3 import HopperEnv
    task=env_config['env_name'];xml,_=paths(task)
    level=str(env_config.get('shift_level','medium'))
    if level not in ['medium','0.5']:raise ValueError('Unsupported v3 shift level: '+level)
    cls=Walker2dEnv if task.startswith('walker2d-') else HopperEnv
    return gym.wrappers.TimeLimit(cls(xml_file=str(xml)),max_episode_steps=1000)

def call_target_dataset(task,shift_level='medium',quality='medium'):
    if quality!='medium':raise ValueError('Only the verified medium dataset is supported')
    if str(shift_level) not in ['medium','0.5']:raise ValueError('Unsupported v3 shift level')
    _,path=paths(task)
    with h5py.File(str(path),'r') as f:
        data={k:f[k][:] for k in ['observations','actions','next_observations','rewards','terminals']}
    assert len({len(v) for v in data.values()})==1
    return dict(observations=data['observations'].astype(np.float32),
                actions=data['actions'].astype(np.float32),
                next_observations=data['next_observations'].astype(np.float32),
                rewards=data['rewards'].astype(np.float32).reshape(-1),
                terminals=data['terminals'].astype(bool).reshape(-1))

def patch_normalization(task,shift_level='medium'):
    import envs.infos as info
    const=normalization(task)
    for key in [task,task+'-'+str(shift_level)]:
        info.REF_MIN_SCORE[key]=const['min'];info.REF_MAX_SCORE[key]=const['max']
    return const
