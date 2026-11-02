"""CPU-only data, loss, routing and short-update equivalence checks; no rollout."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
from support_v4 import *
from methods_v4 import build, IndexedReplayBuffer, SourceBackup, OracleW1IQL
from methods_v3 import TargetIQL
from offline_offline.iql import IQL
from data_v4 import load_source_cpu, verify_source, single_source_dataset, verify_single_buffer
from instrument_v4 import install_loss_trace
from task_env import call_target_dataset, patch_normalization
import argparse
import ast
import numpy as np
import torch


def buffer(data):
    b = IndexedReplayBuffer(17,6,'cpu',max_size=1)
    b.convert_D4RL(data)
    return b


def distribution(x):
    x = np.asarray(x,dtype=np.float64)
    assert np.isfinite(x).all()
    return dict(n=int(x.size),mean=float(x.mean()),std=float(x.std()),
                **{'p'+str(p):float(np.percentile(x,p)) for p in (0,50,90,95,99,100)})


def cpu_steps(model, src, tar, internal_batch=None):
    trace = install_loss_trace(model)
    src.draw_count = tar.draw_count = 0
    np.random.seed(0); torch.manual_seed(0)
    for step in range(10):
        if internal_batch is None:
            model.train(src,tar,128)
        else:
            IQL.train(model,src,tar,internal_batch)
    return dict(trace=trace,hash=model_hash(model),draws=[src.draw_count,tar.draw_count])


def main():
    p = argparse.ArgumentParser(); p.add_argument('--out',type=Path,default=OUT/'cpu1'); a=p.parse_args()
    dest=a.out.resolve(); assert OUT in dest.parents; fresh(dest)
    torch.set_num_threads(2)
    for path in HERE.glob('*.py'):
        ast.parse(path.read_text(encoding='utf-8'))
    protocol=read(OUT/'protocol.json')
    assert protocol['oracle_receipt_sha256']==sha(oracle_receipt())
    old_gate=read(V3_OUT/'stage_a_completed.json')
    for path,h in old_gate['code_sha256'].items():
        assert sha(path)==h
    print('Frozen v3 code and cache receipt verified',flush=True)
    source=load_source_cpu(); src=buffer(source); verify_source(src,oracle_receipt())
    provider=SourceBackup('cpu',oracle_receipt())
    for key,arr in provider.arrays.items():
        assert np.isfinite(arr).all(),key
    assert set(np.unique(provider.arrays['not_done'])) <= {0,1}
    routed={}
    aligned={}
    for method in SINGLE_SOURCE:
        routed[method]=single_source_dataset(src,method,provider)
        aligned[method]=verify_single_buffer(src,buffer(routed[method]),method,provider)
    delta=provider.arrays['next_state'].astype(np.float64)-src.next_state
    changes=int(np.count_nonzero(src.not_done!=provider.arrays['not_done']))
    assert changes==17
    diagnostics=dict(n=src.size,not_done_changed_rows=changes,
        next_state_row_l2=distribution(np.linalg.norm(delta,axis=1)),
        next_state_row_max_abs=distribution(np.abs(delta).max(axis=1)),
        next_state_component_abs=distribution(np.abs(delta).reshape(-1)),
        reward_abs=distribution(np.abs(provider.arrays['reward']-src.reward)),
        all_source_rows_hash_verified=True,all_cache_arrays_sha256_verified=True,
        row_mapping_exact=True,source_only_alignment=aligned)
    write(dest/'data_verified.json',diagnostics)
    del delta
    print('All 999322 rows aligned; terminal changes:',changes,flush=True)
    target=call_target_dataset('walker2d-morph')
    cfg=read(V3_OUT/'s1/t0/effective_config.json')['config']
    initial={}; results={}; references={}
    for method in ALL_METHODS:
        config=dict(cfg,target_only=method!='ORACLE-W1',source_budget=1. if method=='ORACLE-W1' else .2)
        torch.manual_seed(0)
        model=build(method,config,'cpu',oracle_receipt())
        initial[method]=model_hash(model)
        data=routed[method] if method in SINGLE_SOURCE else target
        src_test=buffer(source); tar_test=buffer(data)
        result=cpu_steps(model,src_test,tar_test)
        expected=(0,1280) if method in ('TARGET',)+SINGLE_SOURCE else ((0,640) if method=='TARGET-64' else (640,640))
        assert result['draws']==list(expected)
        assert all(np.isfinite(e['value']) for e in result['trace']) and len(result['trace'])==30
        if method!='ORACLE-W1':
            torch.manual_seed(0)
            ref=TargetIQL(config,'cpu')
            direct=cpu_steps(ref,buffer(source),buffer(data),32 if method=='TARGET-64' else 64)
            assert result==direct, 'Original IQL update equivalence failed: '+method
            references[method]=direct
        else:
            assert model._q_calls==2
        results[method]=result
    assert len(set(initial.values()))==1
    assert initial['TARGET']==read(V3_OUT/'s1/t0/initialization.json')['sha256']
    write(dest/'target_reference_losses_cpu.json',dict(events=references['TARGET']['trace'],
        scope='New CPU replay of frozen v3 TargetIQL for 10 updates; not a historical GPU trace',
        v4_cpu_bitwise_equal=True,initialization_sha256=initial['TARGET'],config=cfg))

    # Independent forward formulas verify which Q call receives oracle labels.
    torch.manual_seed(0)
    w1=build('ORACLE-W1',dict(cfg,target_only=False,source_budget=1.),'cpu',oracle_receipt())
    src.draw_count=0; np.random.seed(5); sample=src.sample(64)
    s,act,ns,r,nd=sample; w1.source_buffer=src; w1.source_backup.buffer=src
    weights=w1.compute_source_weight(*sample)['weight']; assert torch.equal(weights,torch.ones_like(r))
    with torch.no_grad():
        q1,q2=w1.q_funcs(s,act)
        ons,orr,ond=w1.source_backup(src.last_indices)
        y=orr+.99*ond*w1.v_func(ons)
        expected=(q1-y).square()+(q2-y).square()
        actual=w1.q_loss_per_sample(s,act,r,ns,nd)[0]
        assert torch.equal(actual,expected)
        normal_y=r+.99*nd*w1.v_func(ns)
        assert torch.equal(w1.q_loss_per_sample(s,act,r,ns,nd)[0],(q1-normal_y).square()+(q2-normal_y).square())
        try:w1.q_loss_per_sample(s,act,r,ns,nd)
        except RuntimeError:pass
        else:raise AssertionError('Third Q call should fail')
        losses=torch.arange(64,dtype=torch.float32).reshape(-1,1)
        assert torch.equal(w1.combine_target_source_loss(losses,losses+1,weights),losses.mean()+(losses+1).mean())
    class Writer:
        def __init__(self):self.values={}
        def add_scalar(self,tag,value,step):self.values[tag]=float(value)
    writer=Writer(); w1.total_it=1; w1.log_weight_stats(dict(weight=weights),writer)
    assert writer.values['train/source_weight_mean']==1.
    const=patch_normalization('walker2d-morph')
    assert const['min']==10.079455055289959 and const['max']==4398.43
    for kind in ('smoke','formal'):
        try:check_approval(kind)
        except RuntimeError:pass
        else:raise AssertionError('Unapproved GPU run allowed')
    assert not torch.cuda.is_initialized()
    write(dest/'completed.json',dict(passed=True,device='cpu',cuda_initialized=False,
        cpu_update_equivalence_steps_per_path=10,total_cpu_update_calls=90,
        purpose='Contract tests on real arrays; not smoke, rollout, formal experiment or performance evidence',
        initialization_sha256=initial,early_losses_and_post_update_parameters_bitwise_equal=True,
        methods=results,source_q_formula_exact=True,target_q_unchanged=True,q_call_order_enforced=True,
        source_weight_mean=writer.values['train/source_weight_mean'],approval_gates_enforced=True,
        full_pipeline_gpu_reuse_check='pending user authorization',
        protected_files_unchanged=protected_check()))
    print('CPU PREPARATION PASSED; GPU smoke and TARGET reuse remain pending',flush=True)


if __name__=='__main__':
    main()
