"""Read-only verified inputs and frozen v13 solver access."""
import ast
import hashlib
import importlib.metadata as metadata
import os
from pathlib import Path
import sys

import h5py
import numpy as np

from core import HERE,TASK,VARIANTS,SEEDS,read,sha,write,module,segments,choose_episodes

PROJECT = Path.home()/'cluster_zip_reupload_20261104/off-dynamics-transfer-20261104'
V13 = Path.home()/'v13_action_fix'
CODE = V13/'stage0'
RUN = V13/'outputs/stage1_20261110_shared60'
EXPECTED_RECEIPT = {
    'REPORT_ZH.md':'70f32fe2b7e1d0e5d4f3949a3e62578d6804ca1473f5aa223992bbf3256cabf8',
    'RESULT.json':'15afa57c24285ab9ba5381e84cc48e54a75988c7b4ee70dabfd9ce728daf8877',
    'config.json':'d72bc292fad73b6cfaf6f07afa188790cde50c218938801893081aff54747777'}


def tree(root):
    root = Path(root).resolve()
    result = {}
    for p in sorted(root.rglob('*')):
        if p.is_file():
            assert root in p.resolve().parents, 'External symlink: '+str(p)
            result[str(p)] = sha(p)
    return result


def verify_code():
    for name,digest in read(HERE/'FROZEN.json')['files'].items():
        assert sha(HERE/name)==digest, ('V14 package hash mismatch',name)


def inventory():
    """Hash every protected transfer input, used v13 model/cache/log, and environment metadata."""
    verify_code()
    receipt = read(RUN/'STAGE1_COMPLETED.json')
    assert receipt['completed'] and receipt['CPU_audited_runs']==25 and not receipt['stage2_started']
    assert all(receipt['evidence_sha256'][k]==v for k,v in EXPECTED_RECEIPT.items())
    hashes = tree(CODE)
    stage1_cfg = read(RUN/'config.json')
    assert sha(CODE/'FROZEN.json')==stage1_cfg['original_stage0_freeze_sha256']
    for name,digest in read(CODE/'FROZEN.json')['files'].items():
        assert sha(CODE/name)==digest, ('Changed v13 solver package',name)
    for relative,digest in read(HERE/'V13_CODE_EXPECTED.json').items():
        assert sha(V13/relative)==digest, ('Changed v13 code input',relative)
    manifest = read(PROJECT/'MANIFEST.json')
    hashes[str(PROJECT/'MANIFEST.json')] = sha(PROJECT/'MANIFEST.json')
    for e in manifest['files']:
        p = (PROJECT/e['path']).resolve()
        assert PROJECT.resolve() in p.parents and p.stat().st_size==e['bytes'] and sha(p)==e['sha256'], str(p)
        hashes[str(p)] = e['sha256']
    for name,digest in receipt['evidence_sha256'].items():
        p = RUN/name
        assert sha(p)==digest
        hashes[str(p)] = digest
    hashes[str(RUN/'STAGE1_COMPLETED.json')] = sha(RUN/'STAGE1_COMPLETED.json')
    cache = Path(receipt['accepted_cache'])
    assert sha(cache/'CACHE_ACCEPTANCE.json')==receipt['cache_acceptance_sha256']
    cr = read(cache/'CACHE_ACCEPTANCE.json')
    assert cr['passed'] and cr['rows']==999998
    for name,digest in cr['cache_sha256'].items():
        assert sha(cache/name)==digest, name
    hashes.update(tree(cache))
    all_runs = read(RUN/'RESULT.json')['runs']
    assert len(all_runs)==25
    for r in all_runs.values():
        folder = Path(r['folder'])
        assert V13.resolve() in folder.resolve().parents
        done,audit = read(folder/'COMPLETED.json'),read(folder/'AUDIT.json')
        assert audit['passed'] and done['steps']==200000
        assert sha(folder/'final_checkpoint.pt')==done['checkpoint_sha256']==audit['checkpoint_sha256']
        assert sha(folder/'steps.f32')==done['steps_sha256']
        assert sha(folder/'COMPLETED.json')==audit['completed_sha256']
        for name in ('final_checkpoint.pt','steps.f32','COMPLETED.json','AUDIT.json','config.json','JOB.json','INITIALIZATION.json'):
            hashes[str(folder/name)] = sha(folder/name)
    for dirname in ('stage1','stage1_shared60_20261110'):
        for p in (V13/dirname).glob('*'):
            if p.is_file():
                hashes[str(p)] = sha(p)
    provenance = read(CODE/'references/V9_PROVENANCE.json')['accepted_recovery_file_sha256']
    for name,digest in provenance.items():
        p = V13/'stage1/restoration'/Path(name).name
        assert sha(p)==digest
        hashes[str(p)] = digest
    prefix = Path(sys.prefix)
    for package in ('mujoco-py','gym'):
        binding = Path(metadata.distribution(package).locate_file('mujoco_py' if package=='mujoco-py' else 'gym'))
        hashes.update(tree(binding))
        if package=='gym':
            for name in ('envs/mujoco/hopper_v3.py','envs/mujoco/mujoco_env.py','wrappers/time_limit.py','utils/seeding.py','utils/ezpickle.py'):
                assert sha(binding/name)==sha(PROJECT/'environment/gym_source_reference'/name), ('Installed Gym differs from accepted code',name)
    hashes.update(tree(prefix/'conda-meta'))
    native = Path(os.environ['MUJOCO_PY_MUJOCO_PATH'])/'bin/libmujoco210.so'
    hashes[str(native)] = sha(native)
    return hashes


class Inputs:
    def __init__(self):
        self.config = read(HERE/'config.json')
        self.old = read(CODE/'config.json')
        self.cache = Path(read(RUN/'STAGE1_COMPLETED.json')['accepted_cache'])
        self.tasks = read(PROJECT/'cluster/paths.example.json')['tasks']
        self.spec = self.tasks[TASK]
        self.xml = PROJECT/self.spec['xml']
        self.data = {}
        for domain in ('source','target'):
            with h5py.File(PROJECT/self.spec[domain+'_dataset'],'r') as f:
                self.data[domain] = {k:np.asarray(f[k]) for k in ('observations','actions','next_observations','rewards','terminals')}
                if 'timeouts' in f:
                    self.data[domain]['timeouts'] = np.asarray(f['timeouts'])
            assert all(np.isfinite(a).all() for a in self.data[domain].values())
            assert np.isin(self.data[domain]['terminals'],[0,1]).all()
        assert len(self.data['target']['observations'])==5000
        self.raw_ids = self.array('row_id').astype(np.int64)
        assert len(self.raw_ids)==999998 and np.all(self.raw_ids[1:]>self.raw_ids[:-1])
        self.scale = self.array('target_scale')
        expected_scale = self.data['target']['observations'].astype(np.float64).std(axis=0,ddof=1)+self.old['std_epsilon']
        assert np.array_equal(self.scale,expected_scale)
        assert np.array_equal(self.data['source']['observations'][self.raw_ids].astype('float32'),self.array('source_observations'))
        self.target_segments,self.target_steps = segments(self.data['target'])
        self.source_segments,self.source_steps = segments(self.data['source'],source=True)
        assert [r['first_row'] for r in self.target_segments]==[0,652,1320,1963,2618,3265,3911,4564]
        selected,self.complete_source_count = choose_episodes(self.source_segments,500,self.config['B_source_seed'])
        self.selected_source = selected
        seed = self.config['A_source_sample_seed']
        self.fit_positions = np.random.RandomState(seed).choice(len(self.raw_ids),20000,replace=False)
        self.nn_positions = np.random.RandomState(self.config['A_neighbor_seed']).choice(len(self.raw_ids),100000,replace=False)
        self.runs = read(RUN/'RESULT.json')['runs']
        self.solver = module(CODE/'solver.py','v14_frozen_v13_solver').solve
        self.legacy = module(CODE/'legacy_replay.py','v14_frozen_restoration_loader')
        self.restoration = self.legacy.load_restoration(CODE/'references',read(CODE/'RESTORATION_AST.json'))

    def array(self, name):
        return np.load(self.cache/(name+'.npy'),allow_pickle=False,mmap_mode='r')

    def folder(self, variant, seed):
        return Path(self.runs[f'{variant}_s{seed}']['folder'])

    def freeze_plan(self, out):
        write(out/'config.json',dict(self.config,solver_config_original_path=str(CODE/'config.json'),
            v13_config_sha256=sha(CODE/'config.json'),solver=self.old['solver'],
            std_epsilon=self.old['std_epsilon'],target_scale=self.scale.tolist(),
            scale_sha256=sha(self.cache/'target_scale.npy'),
            B0_raw_linf_limit=self.old['target_control_acceptance']['corrected_observation_linf'],
            selected_B_source_trajectories=self.selected_source,source_complete_available=self.complete_source_count,
            source_all_used_if_less_than_500=self.complete_source_count<500,
            target_segments=self.target_segments,source_segments=self.source_segments,
            A_source_sample_positions=self.fit_positions.tolist(),A_neighbor_positions=self.nn_positions.tolist()))

    def new_env(self):
        import gym
        from gym.envs.mujoco.hopper_v3 import HopperEnv
        env = HopperEnv(xml_file=str(self.xml))
        assert env.frame_skip==4 and abs(float(env.model.opt.timestep)-.002)<1e-12
        return gym.wrappers.TimeLimit(env,max_episode_steps=1000)


def policy_class():
    import torch
    path = PROJECT/'source_snapshot/algo/offline_offline/iql_unc.py'
    nodes = [n for n in ast.parse(path.read_text(encoding='utf-8-sig')).body
             if isinstance(n,ast.ClassDef) and n.name in ('MLPNetwork','Policy')]
    assert len(nodes)==2
    ns = dict(torch=torch,nn=torch.nn)
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),ns)
    return ns['Policy']


def load_policies(inputs):
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(inputs.config['model_loading_seed'])
    cls = policy_class()
    result = {}
    for variant in VARIANTS:
        for seed in SEEDS:
            ck = torch.load(str(inputs.folder(variant,seed)/'final_checkpoint.pt'),map_location='cpu')
            assert ck['step']==200000 and ck['job']['variant']==variant and ck['job']['seed']==seed
            policy = cls(11,3,1.,256).to('cpu')
            policy.load_state_dict(ck['policy'],strict=True)
            assert all(torch.isfinite(t).all() for t in policy.state_dict().values())
            digest = hashlib.sha256(b''.join(t.numpy().tobytes() for t in policy.state_dict().values())).hexdigest()
            assert digest==read(inputs.folder(variant,seed)/'COMPLETED.json')['model_digest']
            policy.eval()
            for p in policy.parameters():
                p.requires_grad_(False)
            result[(variant,seed)] = policy
    assert not torch.cuda.is_initialized()
    return result


def actions(policy, obs):
    import torch
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(obs).copy(),dtype=torch.float32,device='cpu')
        value = torch.tanh(policy.network(x).chunk(2,dim=1)[0]).numpy()
    assert np.isfinite(value).all()
    return value
