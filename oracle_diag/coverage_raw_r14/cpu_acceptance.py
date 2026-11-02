"""CPU engineering checks; no real-data optimization, no GPU, no simulator stepping."""
from runtime import *
import numpy as np
import torch
import runpy
import ast
import inspect
import textwrap
import types
import h5py
from learner import initialize,Sampler,Learner
from metrics import group,paired,sd_ratio,best_weight

def main():
    configure_cpu();check_frozen()
    jobs=read(OUT/'training_plan.json')['jobs'];assert len(jobs)==40
    for job in jobs:validate_spec(job)
    for change in [dict(arm='G'),dict(source_w=.5),dict(seed=211),dict(temp=3),dict(steps=1000000),dict(task='ant-morph')]:
        modified=dict(jobs[0],**change)
        try:validate_spec(modified)
        except AssertionError:pass
        else:raise AssertionError(change)
    data_receipts={}
    import mujoco_py
    for spec in protocol()['tasks']:
        task=spec['task'];target=target_data(task);source,ids=source_data(task)
        physical=mujoco_py.load_model_from_path(spec['xml'])
        assert abs(physical.opt.timestep-.002)<1e-12 and physical.nq+physical.nv-1==spec['state_dim']
        with h5py.File(spec['source_dataset'],'r') as file:
            adjacent=np.asarray(file['observations'],np.float32)[ids+1]
        different=np.any(source[2]!=adjacent,axis=1)
        assert not np.any(different & (source[4].reshape(-1)>0))
        assert np.array_equal((source[2]*source[4]),(adjacent*source[4]))
        first=Sampler(target,201,'target_sample','cpu',task=task)
        for _ in range(4):first.sample()
        assert first.hash.hexdigest()==stream_hash(201,'target_sample',5000,4,task)
        data_receipts[task]=dict(target_rows=len(target[0]),source_rows=len(ids),xml_sha256=sha(spec['xml']),
            original_labels=True,oracle_loaded=False,explicit_next_vs_adjacent_differ_only_at_masked_terminals=int(different.sum()),
            score_min=spec['score_min'],score_max=spec['score_max'],model_loaded_without_steps=True,
            canonical_raw_ids_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest())
    hopper=task_spec('hopper-morph');assert hopper['score_min']==-26.3360015397715 and hopper['score_max']==3152.75
    # The old step body differs only at the two source-loss multipliers.
    old=runpy.run_path(str(ROOT/'oracle_diag/coverage_oracle_r13/learner.py'))
    original=inspect.getsource(old['Learner'].step)
    changed=inspect.getsource(Learner.step).replace('self.source_loss_weight * sv.mean()','sv.mean()').replace('self.source_loss_weight * sq.mean()','sq.mean()')
    assert ast.dump(ast.parse(textwrap.dedent(original)))==ast.dump(ast.parse(textwrap.dedent(changed)))
    rng=np.random.RandomState(123)
    def batch():
        return tuple(torch.tensor(a,dtype=torch.float32) for a in (rng.randn(64,17),rng.randn(64,6),rng.randn(64,17),rng.randn(64,1),np.ones((64,1))))
    updates=0
    for w in (0.,1.):
        config=model_config(10,500000,201,'walker2d-morph',w)
        current=initialize(config,'cpu');reference=old['initialize'](config,'cpu')
        assert current.digest()==reference.digest()
        for _ in range(3):
            t,s=batch(),batch();a=current.step(t,None if w==0 else s);b=reference.step(t,None if w==0 else s);updates+=2
            assert all(torch.equal(x,y) for x,y in zip(a,b)) and current.digest()==reference.digest()
    quarter=initialize(model_config(10,500000,201,'walker2d-morph',.25),'cpu')
    t,s=batch(),batch();expected=(quarter.v_loss_per_sample(t[0],t[1])[0].mean()+.25*quarter.v_loss_per_sample(s[0],s[1])[0].mean()).detach()
    got=quarter.step(t,s);updates+=1;assert torch.equal(got[0],expected) and quarter.finite()
    for receipt in read(OUT/'reuse_receipts.json')['runs']:
        if receipt['w']!=0:continue
        initial=initialize(model_config(10,500000,receipt['seed'],'walker2d-morph',0.),'cpu').digest()
        path=Path(receipt['out']);assert initial==read(path/'initialization.json')['sha256']
        assert stream_hash(receipt['seed'],'target_sample',5000,500000,'walker2d-morph')==read(path/'sampling.json')['target_stream_sha256']
    evaluate=runpy.run_path(str(HERE/'train.py'))['evaluate']
    for task in ('walker2d-morph','hopper-morph'):
        spec=task_spec(task);klass=type('Walker2dEnv' if task=='walker2d-morph' else 'HopperEnv',(),{})
        fake=types.SimpleNamespace(_max_episode_steps=1000,unwrapped=klass())
        fake.unwrapped.model=types.SimpleNamespace(opt=types.SimpleNamespace(timestep=.002))
        fake.seed=lambda s:None;fake.reset=lambda:np.zeros(spec['state_dim'])
        fake.step=lambda a:(np.zeros(spec['state_dim']),1.,True,{})
        for run_seed in (201,205 if task=='walker2d-morph' else 210):
            result=evaluate(types.SimpleNamespace(deterministic_action=lambda s:np.zeros(spec['action_dim'])),fake,dict(task=task,seed=run_seed))
            assert result['env_seeds']==list(range(500000+10*(run_seed-201),500010+10*(run_seed-201)))
    assert group([0,1,2,3,4],list(range(5)))['lower_quartile']==1.
    assert paired([1,2,3],[0,0,0],[1,2,3])['positive']==3
    assert sd_ratio([1,2,3,4,5],[1,2,3,4,5],'acceptance',100)['bootstrap_percentile_interval']==[1.,1.]
    assert sd_ratio([1,2,3],[0,0,0],'zero',100)['point'] is None
    assert best_weight({str(w):{'mean':1} for w in (0.,.25,1.)})['selected_w']==0.
    assert not torch.cuda.is_initialized() and not (OUT/'runs').exists()
    write(OUT/'cpu_acceptance.json',dict(passed=True,CPU_only=True,CUDA_initialized=False,
        synthetic_CPU_optimizer_updates=updates,real_data_optimizer_updates=0,simulator_steps=0,
        w1_and_target_update_bitwise_match_R13=True,only_two_source_loss_multipliers_changed=True,
        walker_R13_T_initialization_and_full_target_streams_match=True,eval_namespace_mock_checked=True,
        forty_specs_and_statistics_checked=True,original_dataset_and_XML=data_receipts,time=time.time()))
    write(OUT/'CPU_ACCEPTANCE_COMMITMENT.json',dict(cpu_acceptance_sha256=sha(OUT/'cpu_acceptance.json'),
        code_frozen_sha256=sha(OUT/'code_frozen.json'),protocol_sha256=sha(OUT/'protocol.json'),time=time.time()))
    print('CPU acceptance passed: no CUDA, no environment steps, no real-data fitting',flush=True)

if __name__=='__main__':main()
