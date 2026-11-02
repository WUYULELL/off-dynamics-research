# -*- coding: utf-8 -*-
"""CPU-only acceptance for the actual mixed-source checkpoint and both samplers."""
from runtime import *
import ast
import copy
import numpy as np
import torch
from learner import initialize, Sampler
from resume_state import restore_training, advance_sampler, tensor_equal
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def main():
    configure_cpu(); check_frozen()
    assert not torch.cuda.is_initialized()
    for f in HERE.glob('*.py'):
        ast.parse(f.read_text(encoding='utf-8'))
    old_code = ROOT / 'oracle_diag/coverage_oracle_r13'
    for n in ('learner.py', 'audit.py', 'train.py', 'decision.py'):
        assert (HERE / n).read_bytes() == (old_code / n).read_bytes()
    for name in ('seed', 'configure_cpu', 'env', 'target_data', 'source_data', 'model_config'):
        def function(path):
            return next(n for n in ast.parse(path.read_text(encoding='utf-8')).body
                        if isinstance(n, ast.FunctionDef) and n.name == name)
        assert ast.dump(function(HERE / 'runtime.py')) == ast.dump(function(old_code / 'runtime.py'))
    jobs = read(OUT / 'training_plan.json')['jobs']
    for j in jobs:
        validate_spec(j)
    for change in (dict(arm='G'), dict(arm='T'), dict(arm='U'), dict(arm='C'), dict(seed=201), dict(seed=101),
                   dict(steps=1000000), dict(temp=3), dict(ratio=.2), dict(task='hopper-morph')):
        j = dict(jobs[0]); j.update(change)
        try:
            validate_spec(j)
        except AssertionError:
            pass
        else:
            raise AssertionError(change)
    rec = read(OUT / 'recovery.json')
    original = Path(rec['original_run'])
    assert sha(rec['checkpoint']) == rec['checkpoint_sha256']
    config = read(original / 'config.json')['learner']
    assert config == model_config(10., 500000, 203)
    tar = target_data()
    raw, oracle, raw_ids, meta = source_data()
    mask = np.load(read(OUT / 'R_mask_receipts.json')['masks']['203']['path'])
    assert len(mask) == 49966 and len(raw_ids) == 999322
    for purpose, arrays, ids in (('target_sample', tar, None), ('source_sample', oracle, mask)):
        a = Sampler(arrays, 203, purpose, 'cpu', ids)
        b = Sampler(arrays, 203, purpose, 'cpu', ids)
        for _ in range(200):
            a.sample()
        advance_sampler(b, 200)
        assert a.hash.hexdigest() == b.hash.hexdigest() and tensor_equal(a.rng.get_state(), b.rng.get_state())
    streams = {}
    for purpose, arrays, ids in (('target_sample', tar, None), ('source_sample', oracle, mask)):
        s = Sampler(arrays, 203, purpose, 'cpu', ids)
        advance_sampler(s, 100000)
        first = s.hash.hexdigest()
        assert first == rec['expected_streams'][purpose]['100000']['sha256']
        streams[purpose] = {'100000': first}
    model = initialize(config, 'cpu')
    assert model.digest() == read(original / 'initialization.json')['sha256']
    saved = restore_training(model, rec['checkpoint'])
    assert model.finite() and model.total_it == 100000
    def all_finite(x):
        if torch.is_tensor(x):
            return bool(torch.isfinite(x).all())
        if isinstance(x, dict):
            return all(all_finite(v) for v in x.values())
        if isinstance(x, (list, tuple)):
            return all(all_finite(v) for v in x)
        return True
    assert all_finite(saved)
    ts = Sampler(tar, 203, 'target_sample', 'cpu')
    ss = Sampler(oracle, 203, 'source_sample', 'cpu', mask)
    advance_sampler(ts, 100000); advance_sampler(ss, 100000)
    rng = torch.get_rng_state().clone()
    for _ in range(20):
        model.step(ts.sample(), ss.sample())
    assert torch.equal(rng, torch.get_rng_state()), 'Mixed update unexpectedly consumes Torch RNG'
    test_path = OUT / 'CPU_split_checkpoint100020.pt'
    torch.save(model.checkpoint(), str(test_path))
    restored = initialize(config, 'cpu')
    restore_training(restored, test_path)
    rt = Sampler(tar, 203, 'target_sample', 'cpu')
    rs = Sampler(oracle, 203, 'source_sample', 'cpu', mask)
    advance_sampler(rt, 100020); advance_sampler(rs, 100020)
    for _ in range(20):
        left = model.step(ts.sample(), ss.sample())
        right = restored.step(rt.sample(), rs.sample())
        assert all(torch.equal(x, y) for x, y in zip(left, right))
    assert tensor_equal(model.checkpoint(), restored.checkpoint())
    assert ts.hash.hexdigest() == rt.hash.hexdigest() and ss.hash.hexdigest() == rs.hash.hexdigest()
    ea = EventAccumulator(str(Path(shortpath(original)) / 'tb'), size_guidance={'scalars': 0}).Reload()
    assert all(not [v for v in ea.Scalars(tag) if v.step > 100000]
               for tag in ('loss/V', 'loss/Q', 'loss/actor'))
    write(OUT / 'RESTORE_ACCEPTANCE_EXPECTATIONS.json', dict(
        checkpoint_sha256=rec['checkpoint_sha256'], checkpoint_step=100000,
        GPU_loss_overlap_available=False,
        reason='User-requested stop immediately after first checkpoint; no old post-checkpoint GPU losses exist',
        GPU_exact_full_state_roundtrip_required=True,
        no_new_GPU_gradient_probe_updates=True))
    for item in read(OUT / 'reuse_receipts.json')['runs']:
        a = read(item['audit']); done = read(Path(item['job']['out']) / 'completed.json')
        assert a['passed'] and sha(item['audit']) == item['audit_sha256']
        assert sha(Path(item['job']['out']) / 'completed.json') == a['completed_sha256']
        assert sha(Path(item['job']['out']) / 'evaluations.json') == a['evaluations_sha256']
        assert sha(done['final_model']) == a['final_model_sha256'] == done['final_model_sha256']
    for s, item in read(OUT / 'R_mask_receipts.json')['masks'].items():
        assert sha(item['path']) == item['sha256']
        assert np.array_equal(np.load(item['path']), np.load(OUT / 'random_masks' / ('R_%s.npy' % s)))
        assert item['draws_of_candidate_masks'] == 1 and not item['selected_using_returns']
    assert not torch.cuda.is_initialized() and not (OUT / 'runs').exists()
    result = dict(passed=True, CPU_only=True, CUDA_initialized=False,
        checkpoint_step=100000, full_state_restored=True, scheduler_Tmax=500000,
        CPU_split_resume_bitwise_identical=True, CPU_mixed_update_consumes_Torch_RNG=False,
        both_sampler_loop_vs_chunk_identical=True, stream_hashes=streams,
        original_100k_target_and_source_streams_match=True,
        learner_audit_fresh_R_train_decision_byte_identical=True, scientific_runtime_functions_AST_identical=True,
        reused17_audit_model_and_evaluation_hashes_verified=True, R_masks_unchanged=True,
        GPU_loss_overlap_available=False, GPU_exact_full_state_roundtrip_pending=True,
        long_training_updates=0, real_environment_steps=0, diagnostic_CPU_test_updates=40,
        no_old_writes=True, time=time.time())
    write(OUT / 'cpu_acceptance.json', result)
    write(OUT / 'CPU_ACCEPTANCE_COMMITMENT.json', dict(cpu_acceptance_sha256=sha(OUT / 'cpu_acceptance.json'),
        restore_expectations_sha256=sha(OUT / 'RESTORE_ACCEPTANCE_EXPECTATIONS.json'), code_frozen_sha256=sha(OUT / 'code_frozen.json'), time=time.time()))
    print('CPU mixed-source full-state/split-resume and two sampling streams passed; GPU exact full-state roundtrip pending; no old GPU loss overlap available', flush=True)

if __name__ == '__main__':
    main()
