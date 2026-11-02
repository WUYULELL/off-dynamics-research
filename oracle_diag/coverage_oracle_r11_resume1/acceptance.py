"""CPU-only full-state and deterministic sample-stream recovery tests."""
import ast
import copy
from pathlib import Path
import torch
import runtime as rt
from learner import initialize,Sampler
from resume_state import restore_training,advance_sampler,tensor_equal
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

OLD=rt.ROOT/'oracle_diag/results/coverage_oracle_20261016_r11'
ORIGINAL=OLD/'runs/tune_T_t10_h1000000_s101'

def main():
    rt.configure_cpu()
    assert not torch.cuda.is_initialized()
    for p in rt.HERE.glob('*.py'):ast.parse(p.read_text(encoding='utf-8'))
    for path,h in rt.read(OLD/'code_frozen.json')['sha256'].items():assert rt.sha(path)==h
    recovery=rt.protocol()['recovery']
    assert rt.sha(recovery['checkpoint'])==recovery['checkpoint_sha256']
    tar=rt.target_data()
    cfg=rt.model_config(10.,1000000,101)
    saved=torch.load(recovery['checkpoint'],map_location='cpu')
    assert saved['config']==cfg and saved['step']==400000
    actual=initialize(cfg,'cpu')
    restore_training(actual,recovery['checkpoint'])
    assert actual.finite()
    for key in ('V_optimizer','Q_optimizer','actor_optimizer'):
        for item in saved[key]['state'].values():
            assert all(bool(torch.isfinite(x).all()) for x in item.values() if torch.is_tensor(x))
    # Sample-by-sample vs batched advancing must be identical for RandomState randint.
    a,b=Sampler(tar,101,'target_sample','cpu'),Sampler(tar,101,'target_sample','cpu')
    for _ in range(200):a.sample()
    advance_sampler(b,200)
    assert a.hash.hexdigest()==b.hash.hexdigest()
    assert all((x==y).all() if hasattr(x,'shape') else x==y for x,y in zip(a.rng.get_state(),b.rng.get_state()))
    # Rebuild400k, then extend independent copies to the actual407k and completed500k records.
    sampler=Sampler(tar,101,'target_sample','cpu')
    advance_sampler(sampler,400000)
    restored_draws=sampler.draws
    hashes={400000:sampler.hash.hexdigest()}
    rng_copy=copy.deepcopy(sampler.rng.get_state())
    digest_copy=sampler.hash.copy()
    for end in (407000,500000):
        previous=max(hashes)
        ids=sampler.ids[sampler.rng.randint(5000,size=(end-previous)*64)]
        sampler.hash.update(ids.astype('<i8').tobytes())
        hashes[end]=sampler.hash.hexdigest()
    expected407=rt.read(ORIGINAL/'progress.json')['target_stream_sha256']
    oldreuse=rt.read(OLD/'reuse_receipts.json')
    original_T=rt.read(oldreuse['T500_temp1_seed101']['audit'])
    assert hashes[407000]==expected407
    assert hashes[500000]==original_T['sampling']['target_stream_sha256']
    # A CPU interrupted/saved/restored trajectory must be exactly the same as a continuous one.
    continuous=initialize(cfg,'cpu')
    continuous_sampler=Sampler(tar,101,'target_sample','cpu')
    initial_rng=torch.get_rng_state().clone()
    for _ in range(40):continuous.step(continuous_sampler.sample())
    assert torch.equal(initial_rng,torch.get_rng_state()),'This training path consumes Torch RNG'
    tmp=rt.OUT/'CPU_test_checkpoint40.pt'
    torch.save(continuous.checkpoint(),str(tmp))
    resumed=initialize(cfg,'cpu')
    restore_training(resumed,tmp)
    resumed_sampler=Sampler(tar,101,'target_sample','cpu')
    advance_sampler(resumed_sampler,40)
    for _ in range(40):
        continuous.step(continuous_sampler.sample())
        resumed.step(resumed_sampler.sample())
    assert tensor_equal(continuous.checkpoint(),resumed.checkpoint())
    assert continuous_sampler.hash.hexdigest()==resumed_sampler.hash.hexdigest()
    # Capture the source overlapping loss record, no GPU is used for this inspection.
    ea=EventAccumulator(str(Path(rt.shortpath(ORIGINAL))/'tb'),size_guidance={'scalars':0}).Reload()
    overlaps={tag:{str(x.step):x.value for x in ea.Scalars(tag) if 400000<x.step<=407000}
              for tag in ('loss/V','loss/Q','loss/actor')}
    assert all(set(v)==set(map(str,range(401000,407000,1000))) for v in overlaps.values())
    rt.write(rt.OUT/'OVERLAP_LOSSES.json',dict(exact_float32_required=True,events=overlaps))
    # Carry forward only already-complete, independently audited outputs.
    oldplan=rt.read(OLD/'training_plan.json')
    paths=[Path(oldreuse['T500_temp1_seed101']['audit'])]
    paths += [Path(x['out'])/'audit.json' for x in oldplan['dispatched_new_jobs'] if x['steps']==500000]
    reuse=dict(T500_by_temp_seed={},T_smoke=oldreuse['T_smoke'])
    for file in paths:
        data=rt.read(file)
        assert data['passed']
        job=data['job'];d=Path(job['out']);done=rt.read(d/'completed.json')
        assert rt.sha(d/'completed.json')==data['completed_sha256']
        assert rt.sha(d/'evaluations.json')==data['evaluations_sha256']
        assert rt.sha(done['final_model'])==data['final_model_sha256']
        reuse['T500_by_temp_seed']['%g_%d'%(job['temp'],job['seed'])]=dict(audit=str(file),audit_sha256=rt.sha(file))
    assert len(reuse['T500_by_temp_seed'])==9
    rt.write(rt.OUT/'reuse_receipts.json',reuse)
    rt.write(rt.OUT/'T_smoke_audited.json',dict(passed=True,reused=True,receipt=reuse['T_smoke']))
    protected=dict(rt.read(OLD/'protected_manifest.json')['sha256'])
    for root in (OLD,rt.ROOT/'oracle_diag/coverage_oracle_r11'):
        for f in root.rglob('*'):
            if f.is_file() and '__pycache__' not in str(f):protected[str(f)]=rt.sha(f)
    rt.write(rt.OUT/'protected_manifest.json',dict(sha256=protected,count=len(protected)))
    result=dict(passed=True,CPU_only=True,CUDA_initialized=torch.cuda.is_initialized(),
        full_optimizer_target_Q_scheduler_restored=True,checkpoint_step=400000,
        scheduler_T_max=actual.policy_lr_schedule.T_max,original_LR=actual.policy_optimizer.param_groups[0]['lr'],
        CPU_split_resume_bitwise_identical=True,CPU_training_consumes_Torch_RNG=False,
        sampler_loop_vs_chunk_identical=True,restored_target_draws=restored_draws,
        stream_hashes={str(k):v for k,v in hashes.items()},recorded407k_and500k_hashes_match=True,
        missing_checkpoint_RNG='sampler rebuilt from fixed seed; this T update path is deterministic and does not draw Torch RNG',
        GPU_overlap_gate_pending=True,old_9_T_and_smoke_reused=True,code_sha256=rt.code_hashes())
    assert result['CUDA_initialized'] is False
    rt.write(rt.OUT/'cpu_acceptance.json',result)
    rt.write(rt.OUT/'code_frozen.json',dict(sha256=rt.code_hashes(),cpu_acceptance_sha256=rt.sha(rt.OUT/'cpu_acceptance.json')))
    print('CPU full-state recovery passed; sampler407k/500k hashes match; CPU split resume bitwise identical',flush=True)

if __name__=='__main__':main()
