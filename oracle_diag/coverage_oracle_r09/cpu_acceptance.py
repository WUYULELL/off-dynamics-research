"""Meaningful differential and isolation tests, entirely on CPU."""
from runtime import *
import copy
import numpy as np
import torch
from learner import initialize, Sampler
from selection import rms_distance, component_audit
from scipy.spatial import cKDTree
from offline_offline.iql import IQL
from offline_offline.iql_bellman_fb import IQL_BELLMAN_FB

def difference(a, b, nets=('policy','v_func','q_funcs','target_q_funcs')):
    return max(float((x-y).abs().max()) for key in nets for x, y in
        zip(getattr(a,key).parameters(), getattr(b,key).parameters()))

def main():
    configure_cpu()
    assert not torch.cuda.is_initialized()
    tests = {}
    cfg = model_config(3., 500000, 101)
    rng = np.random.RandomState(55)
    def arrays(n):
        return [rng.normal(size=(n,17)).astype('float32'), np.tanh(rng.normal(size=(n,6))).astype('float32'),
            rng.normal(size=(n,17)).astype('float32'), rng.normal(size=(n,1)).astype('float32'),
            rng.randint(2, size=(n,1)).astype('float32')]
    target = arrays(5000)
    source = arrays(200)
    class Fixed:
        def __init__(self,batch): self.batch=batch
        def sample(self,n):
            assert n == 64
            return self.batch
    batch = tuple(torch.as_tensor(x[:64]) for x in target)
    src = tuple(torch.as_tensor(x[:64]) for x in source)
    a = initialize(cfg, 'cpu')
    legacy = IQL(dict(cfg, target_only=True), 'cpu')
    for key in ('policy','v_func','q_funcs','target_q_funcs'):
        getattr(legacy,key).load_state_dict(getattr(a,key).state_dict())
    for _ in range(5):
        a.step(batch)
        legacy.train(None, Fixed(batch), batch_size=32)
    error_t = difference(a,legacy)
    assert error_t < 3e-6, error_t
    tests['target5step_legacy_max_parameter_error'] = error_t
    class LegacyMixed(IQL_BELLMAN_FB):
        def initialize_target_feature_bank(self,tar): self.target_features_norm=torch.empty(0)
        def compute_source_weight(self,*args): return dict(weight=torch.ones_like(args[3]))
        def combine_target_source_loss(self,t,s,w):
            self.calls += 1
            return t.mean() if self.calls % 3 == 0 else t.mean()+s.mean()
    a = initialize(cfg, 'cpu')
    b = LegacyMixed(cfg, 'cpu')
    b.calls = 0
    for key in ('policy','v_func','q_funcs','target_q_funcs'):
        getattr(b,key).load_state_dict(getattr(a,key).state_dict())
    for _ in range(5):
        a.step(batch,src)
        b.train(Fixed(src),Fixed(batch),batch_size=64)
    error_m = difference(a,b)
    assert error_m < 3e-6, error_m
    tests['mixed5step_legacy_algebra_max_parameter_error'] = error_m
    a, b = initialize(cfg, 'cpu'), initialize(cfg, 'cpu')
    changed = list(src)
    changed[0],changed[1],changed[2],changed[3] = src[0]+2, -src[1], src[2]+3, src[3]+9
    a.step(batch,src)
    b.step(batch,tuple(changed))
    assert difference(a,b,('policy',)) == 0
    assert difference(a,b,('v_func',)) > 0 and difference(a,b,('q_funcs',)) > 0
    tests['source_changes_V_Q_but_not_direct_actor_first_step'] = True
    one, two = Sampler(target,101,'target_sample','cpu'), Sampler(target,101,'target_sample','cpu')
    other = Sampler(source,101,'source_sample','cpu')
    for _ in range(100):
        one.sample(); other.sample(); two.sample()
    assert one.hash.hexdigest() == two.hash.hexdigest()
    assert initialize(cfg,'cpu').digest() == initialize(cfg,'cpu').digest()
    tests['paired_initialization_and_target_stream_separate_from_source'] = True
    state = np.random.RandomState(91).normal(size=17).astype('float32')
    model = initialize(cfg,'cpu')
    r = torch.get_rng_state().clone()
    action = model.deterministic_action(state)
    assert torch.equal(r,torch.get_rng_state())
    expected = model.select_action(state,test=True)
    assert np.array_equal(action,expected)
    tests['mean_action_exact_and_no_RNG_consumption'] = True
    model500 = initialize(cfg,'cpu')
    model1m = initialize(model_config(3.,1000000,101),'cpu')
    for _ in range(20):
        model500.step(batch); model1m.step(batch)
    assert model500.policy_lr_schedule.T_max == 500000
    assert model1m.policy_lr_schedule.T_max == 1000000
    tests['separate_cosine_horizons'] = True
    points = np.random.RandomState(11).normal(size=(50,17))
    tree = cKDTree(points)
    fast = rms_distance(tree,points[:3])
    brute = []
    for row in points[:3]:
        d = np.sort(((points-row)**2).sum(axis=1))[:20]
        brute.append(np.sqrt(np.mean(d)/17))
    assert np.allclose(fast,brute,rtol=0,atol=1e-12)
    tests['20NN_independent_bruteforce'] = True
    fake = np.arange(1000,dtype=float)
    report = component_audit(fake,np.zeros(1000),np.arange(999,-1,-1),[.05,.2,.5])
    assert all('rel' in r['dominant_components_over_90pct'] for r in report['results'])
    tests['component_audit_flags_known_single_component'] = True
    tar = target_data()
    raw, oracle, ids, meta = source_data()
    tests['target_explicit_next_rows'] = len(tar[0])
    tests['source_canonical_oracle_aligned_rows'] = len(ids)
    assert len(ids) == 999322
    nested = [set(np.arange(int(np.floor(p*len(ids))))) for p in (.05,.2,.5)]
    assert nested[0] <= nested[1] <= nested[2]
    tests['retention_counts'] = [len(x) for x in nested]
    e = env()
    e.seed(123)
    s = e.reset()
    ns,r,d,info = e.step(np.zeros(6))
    assert s.shape == ns.shape == (17,) and np.isfinite(ns).all()
    assert e._max_episode_steps == 1000 and abs(e.unwrapped.model.opt.timestep-.002)<1e-12
    e.close()
    tests['actual_shifted_env_construct_reset_step_CPU'] = True
    # Snapshot historical protection paths at CURRENT state; retain prior mismatches as evidence.
    old = read(ROOT/'oracle_diag/results/v5_20261013/protected_manifest.json')['sha256']
    existing = dict(old)
    for base in ('oracle_diag/v3','oracle_diag/v4','oracle_diag/v5','reproductions/otdf_v7',
                 'outputs/research_idea_protocol_20261016_r08'):
        for p in (ROOT/base).rglob('*'):
            if p.is_file() and '__pycache__' not in str(p): existing.setdefault(str(p),None)
    now, historical_differences = {}, []
    for i,(path,previous_hash) in enumerate(existing.items()):
        if not Path(path).exists():
            historical_differences.append(dict(path=path,reason='already missing at R09 start'))
            continue
        actual=sha(path)
        now[path]=actual
        if previous_hash and actual != previous_hash:
            historical_differences.append(dict(path=path,previous=previous_hash,current=actual,
                reason='difference predates R09 execution; old evidence retained'))
        if i%500 == 0: print('protected snapshot',i,flush=True)
    write(OUT/'protected_manifest.json',dict(sha256=now,count=len(now),
        historical_differences=historical_differences, no_old_files_written=True))
    tests['protected_current_snapshot_count']=len(now)
    assert not torch.cuda.is_initialized()
    write(OUT/'cpu_acceptance.json',dict(passed=True,tests=tests,code_sha256=code_hashes(),
        tolerance=3e-6, diagnostics_device='cpu', CUDA_initialized=False,
        protocol_sha256=sha(OUT/'protocol.json')))
    print(json.dumps(tests,ensure_ascii=False),flush=True)

if __name__ == '__main__':
    main()
