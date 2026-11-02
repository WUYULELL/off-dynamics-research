"""Eight synthetic learner updates only; run after Walker completes, before GPU trials."""
from common import *
import numpy as np
import torch
from actor import initialize, source_weights
from learner import initialize as original_initialize

def main():
    guard(); r.configure_cpu()
    assert not torch.cuda.is_initialized()
    c = settings()['learner']
    rng = np.random.RandomState(20261101)
    batches = []
    for _ in range(4):
        batches.append(tuple(torch.as_tensor(a) for a in (
            rng.normal(size=(64,11)).astype('float32'), rng.uniform(-1,1,size=(64,3)).astype('float32'),
            rng.normal(size=(64,11)).astype('float32'), rng.normal(size=(64,1)).astype('float32'),
            rng.randint(0,2,size=(64,1)).astype('float32'))))
    original = original_initialize(c, 'cpu')
    zero = initialize(c, 'cpu', 'SA-all', 0.)
    all_model = initialize(c, 'cpu', 'SA-all', 1.)
    top_model = initialize(c, 'cpu', 'SA-top', 1.)
    fixture = r.read(r.OUT/'PAIRING_EXPECTATIONS.json')['seeds']['201']['initialization']
    assert original.digest() == zero.digest() == all_model.digest() == top_model.digest() == fixture
    parity = []
    for k in range(2):
        target, source = batches[2*k:2*k+2]
        a, b = original.step(target, source), zero.step(target, source)
        assert all(torch.equal(x,y) for x,y in zip(a,b))
        assert tree_digest(scientific_state(original)) == tree_digest(scientific_state(zero))
        all_model.step(target, source); top_model.step(target, source)
        assert tree_digest(critic_state(original)) == tree_digest(critic_state(all_model)) == tree_digest(critic_state(top_model))
        parity.append(dict(step=k+1, losses=[float(x) for x in a], full_state_bitwise_equal=True))
    # Strict >0, top before positive filter, cap100, stable tie, zero-selection branch.
    a = torch.ones(64,1)
    w, keep = source_weights(a, 'SA-top')
    assert keep.reshape(-1).tolist() == [True]*16+[False]*48 and float(w.sum()) == 1600.
    a = torch.full((64,1), -1.); a[:3] = .1
    w, keep = source_weights(a, 'SA-top'); assert int(keep.sum()) == 3
    assert torch.equal(w[keep], torch.exp(10*a[keep]).clamp(max=100.))
    w, keep = source_weights(torch.zeros(64,1), 'SA-top'); assert not keep.any() and float(w.sum()) == 0.
    assert all(full_finite(scientific_state(m)) for m in (original,zero,all_model,top_model))
    assert all(m.policy_lr_schedule.T_max == 500000 for m in (original,zero,all_model,top_model))
    assert not torch.cuda.is_initialized()
    r.write(OUT/'CPU_ACCEPTANCE.json', dict(passed=True, synthetic_learner_updates=8,
        synthetic_Adam_steps=24, real_data_optimization_updates=0, environment_steps=0,
        CUDA_initialized=False, bitwise_wa0=parity, critics_bitwise_for_all_variants=True,
        mask_boundary_tests_passed=True, initialization_sha256=fixture,
        loss_correction='O1 advantage-weighted action MSE; negative-log-pi superseded', time=time.time()))

if __name__ == '__main__': main()
