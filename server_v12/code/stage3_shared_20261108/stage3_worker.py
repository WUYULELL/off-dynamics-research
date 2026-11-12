"""CPU diagnostics, frozen learners, and CPU audit for V12 stage 3."""
import argparse
import ast
import hashlib
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

from stage3_setup import CAP, EVAL_EVERY, GPU_UUIDS, OUT, SEEDS, STAGE2, STEPS, TASK
from stage3_setup import load_legacy, points, read, sha, source_mode, verify_new


R13 = 'source_snapshot/oracle_diag/coverage_oracle_r13'
R14 = 'source_snapshot/oracle_diag/coverage_raw_r14'


def load_root_data(project, trial):
    import h5py
    import numpy as np
    source, target, ids, receipt = trial.load_pair(project)
    with h5py.File(receipt['source'], 'r') as file:
        adjacent = np.asarray(file['observations'][()], dtype=np.float32)[ids + 1]
    different = (source['next_observations'] != adjacent).any(axis=1)
    terminal = source['terminals'].reshape(-1).astype(bool)
    target_different = (target['next_observations'][:-1] != target['observations'][1:]).any(axis=1)
    target_terminal = target['terminals'][:-1].reshape(-1).astype(bool)
    receipt.update(ROOT_source_next=source_mode(), source_next_different_rows=int(different.sum()),
                   source_next_nonterminal_different_rows=int((different & ~terminal).sum()), target_explicit_next=True,
                   target_next_different_rows=int(target_different.sum()),
                   target_next_nonterminal_different_rows=int((target_different & ~target_terminal).sum()),
                   target_rows_compared=4999, target_explicit_extra_row=4999)
    assert receipt['source_next_different_rows'] == 513 and receipt['source_next_nonterminal_different_rows'] == 0
    assert len(target['observations']) == 5000 and receipt['target_next_different_rows'] == 7
    assert receipt['target_next_nonterminal_different_rows'] == 3
    source['next_observations'] = adjacent
    return source, target, ids, receipt


def root_source_proof(trial):
    path = trial.UPSTREAM / 'algo/ROOT.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    root = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'ROOT')
    train = next(x for x in root.body if isinstance(x, ast.FunctionDef) and x.name == 'train')
    nodes = list(ast.walk(train))
    calls = [n for n in nodes if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
             and isinstance(n.func.value, ast.Name) and n.func.value.id == 'self' and n.func.attr == 'update_policy']
    assert len(calls) == 1 and len(calls[0].args) == 3 and not calls[0].keywords
    assert not any(isinstance(n, ast.Constant) and n.value == 'proportion' for n in nodes)
    assert not any(isinstance(n, ast.Attribute) and n.attr == 'topk' for n in nodes)
    return dict(code_sha256=sha(path), actor_weight_argument_passed=False,
                active_proportion_filter=False, actor_call_line=calls[0].lineno, actor_call='self.update_policy(adv, state, action)')


def solver_seed_proof(trial):
    path = trial.UPSTREAM / 'robust_ot_solver.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'solve_robust_ot', 'filter_dataset', 'compute_and_save_weights'}
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names}
    assert functions.keys() == names
    attrs = set()
    for name, function in functions.items():
        allowed_args = {arg.arg for arg in function.args.args}
        assert 'seed' not in allowed_args, f'{name} accepts a training seed'
        for node in ast.walk(function):
            if isinstance(node, (ast.Attribute, ast.Name)):
                identifier = node.attr if isinstance(node, ast.Attribute) else node.id
                assert identifier not in {'seed', 'random', 'rng', 'PRNGKey', 'default_rng'}, f'{name} uses {identifier}'
            if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) and node.value.id == 'args':
                attrs.add(node.attr)
            if isinstance(node, ast.Name) and node.id == 'args':
                assert name != 'solve_robust_ot'
    assert attrs == {'epsilon', 'lambda_src', 'lambda_tar'}, attrs
    return dict(passed=True, training_seed_input=False, RNG_use_in_pinned_solver=False,
                solver_sha256=sha(path), inspected_functions=sorted(names),
                effective_solver_arguments={name: trial.CLI[name] for name in sorted(attrs)},
                solve_calls_for_formal_stage=1, formal_training_seeds=list(SEEDS))


def compare_weight_distributions(new_path, old_path, new_receipt, old_receipt):
    import h5py
    import numpy as np

    assert new_receipt['source_rows_sha256'] == old_receipt['source_rows_sha256']
    assert new_receipt['rows'] == old_receipt['rows'] == 999322

    def load(path, receipt):
        assert sha(path) == receipt['sha256']
        with h5py.File(path, 'r') as handle:
            raw = handle['cost'][()]
        assert raw.shape == (receipt['rows'],) and np.isfinite(raw).all() and (raw >= 0).all()
        assert raw.mean() > 0
        normalized = (raw / (raw.mean() + 1e-12)).astype(np.float32)
        return dict(raw=raw, normalized_float32=normalized, clipped=np.minimum(normalized, 5))

    def describe(values):
        quantiles = np.quantile(values, [0, .01, .25, .5, .75, .95, .99, 1])
        return dict(mean=float(values.mean()), quantiles=dict(zip(('min', 'p01', 'p25', 'p50', 'p75', 'p95', 'p99', 'max'), quantiles.tolist())))

    def pearson(a, b):
        if a.min() == a.max() or b.min() == b.max():
            return None
        return float(np.corrcoef(a, b)[0, 1])

    current, prior = load(new_path, new_receipt), load(old_path, old_receipt)
    distributions = {stage: {key: describe(array) for key, array in data.items()}
                     for stage, data in (('stage3', current), ('stage2', prior))}
    for stage, data in (('stage3', current), ('stage2', prior)):
        distributions[stage]['fraction_clipped_at_5'] = float(np.mean(data['normalized_float32'] > 5))
        distributions[stage]['fraction_zero'] = float(np.mean(data['raw'] == 0))
        reference = new_receipt if stage == 'stage3' else old_receipt
        assert all(distributions[stage][key] == reference[key] for key in distributions[stage]), stage
    return dict(purpose='descriptive only', aligned_source_rows=new_receipt['rows'],
                aligned_source_rows_sha256=new_receipt['source_rows_sha256'],
                stage3_file_sha256=new_receipt['sha256'], stage2_file_sha256=old_receipt['sha256'],
                distributions=distributions, pearson={k: pearson(current[k], prior[k]) for k in current})


def compare_q(root, snapshot, source, target, weights, torch, rng):
    """Evaluate three losses on the same real, sampled batch without a train call."""
    import numpy as np
    s_ids = rng.randint(len(source['observations']), size=128)
    t_ids = rng.randint(len(target['observations']), size=128)
    def tensor(key):
        src = source[key][s_ids].reshape(len(s_ids), -1)
        dst = target[key][t_ids].reshape(len(t_ids), -1)
        return torch.as_tensor(np.concatenate([src, dst]), dtype=torch.float32)

    s, a, n = (tensor(key) for key in ('observations', 'actions', 'next_observations'))
    r, not_done = tensor('rewards'), 1 - tensor('terminals')
    q = root.DoubleQFunc(17, 6, hidden_size=256)
    v = root.ValueFunc(17, 6, hidden_size=256)
    q.load_state_dict(snapshot['networks']['q_funcs'])
    v.load_state_dict(snapshot['networks']['v_func'])
    q.eval()
    v.eval()
    source_weights = torch.as_tensor((weights[s_ids] / (weights.mean() + 1e-12)).reshape(-1, 1), dtype=torch.float32)
    w = torch.cat([source_weights.clamp(max=5), torch.ones((128, 1))])
    view = SimpleNamespace(weight=w.unsqueeze(1), q_funcs=q, v_func=v, discount=.99, total_it=0)
    with torch.no_grad():
        upstream = float(root.ROOT.update_q_functions(view, s, a, r, n, not_done))
        vt = r + not_done * view.discount * v(n)
        q1, q2 = q(s, a)
        residual1, residual2 = (q1 - vt).square(), (q2 - vt).square()
        assert tuple(residual1.shape) == (256, 1)
        per_sample = float((w * residual1).mean() + (w * residual2).mean())
        unweighted = float(residual1.mean() + residual2.mean())
        assert math.isclose(upstream, float(w.mean()) * unweighted, rel_tol=3e-6, abs_tol=1e-8)
    assert all(math.isfinite(x) and x > 0 for x in (upstream, per_sample, unweighted))
    return dict(losses=dict(upstream=upstream, per_sample_weighted=per_sample, unweighted=unweighted),
                ratios=dict(upstream_over_unweighted=upstream / unweighted, per_sample_over_unweighted=per_sample / unweighted,
                            upstream_over_per_sample=upstream / per_sample),
                upstream_weight_shape=list(view.weight.shape), per_sample_weight_shape=list(w.shape),
                residual_shape=list(residual1.shape), broadcast_shape=[256, 256, 1], mean_weight=float(w.mean()),
                source_batch_rows=128, target_batch_rows=128, canonical_source_batch_indices=s_ids.tolist(),
                target_batch_indices=t_ids.tolist())


def probe(job, legacy):
    bc, common, _, _, trial, science = legacy
    import h5py
    import numpy as np
    import torch
    bc.configure_torch(torch)
    project = science.Project(job['project'])
    source, target, ids, inputs = load_root_data(project, trial)
    complete = read(STAGE2 / 'TRAINING_COMPLETED.json')
    weight_info = read(STAGE2 / 'WEIGHTS_COMPLETED.json')
    ck = STAGE2 / 'final_checkpoint.pt'
    weights = STAGE2 / 'robust_weights.hdf5'
    assert sha(ck) == complete['checkpoint_sha256'] and sha(weights) == weight_info['sha256']
    assert weight_info['source_rows_sha256'] == inputs['source_rows_sha256']
    snapshot = torch.load(str(ck), map_location='cpu')
    assert snapshot['main_steps'] == 20000 and bc.finite_state(torch, snapshot['networks'])
    root = trial.load_upstream('algo/ROOT.py', '_v12_stage3_probe_ROOT')
    proof = root_source_proof(trial)
    solver_proof = solver_seed_proof(trial)
    with h5py.File(weights, 'r') as file:
        cost = file['cost'][()]
    assert cost.shape == (len(ids),) and np.isfinite(cost).all() and (cost >= 0).all() and cost.mean() > 0
    result = compare_q(root, snapshot, source, target, cost, torch, np.random.RandomState(501))
    result['raw_source_batch_rows'] = ids[result['canonical_source_batch_indices']].tolist()
    assert not torch.cuda.is_initialized()
    result.update(passed=True, CPU_only=True, optimizer_updates=0, environment_steps=0, CUDA_initialized=False,
                  batch_selection_seed=501, checkpoint_source='completed stage 2 (seed 901, step 20000)',
                  checkpoint_sha256=sha(ck), weight_source='completed stage 2 explicit-next weights, probe only',
                  weight_sha256=sha(weights), **proof)
    out = Path(job['output'])
    common.write(out / 'ROOT_Q_READONLY_PROBE.json', result)
    common.write(out / 'ROOT_INPUTS.json', inputs)
    common.write(out / 'ROOT_OT_SEED_INDEPENDENCE.json', solver_proof)
    print('CPU Q probe passed:', result['losses'], result['ratios'], flush=True)


def runtime(project, science, common):
    import numpy as np
    import torch
    spec = project.spec(TASK)
    data_path = project.path(spec['target_dataset'])
    spec.update(target_dataset=str(data_path), target_dataset_sha256=sha(data_path))
    namespace = dict(np=np, torch=torch, random=random, hashlib=hashlib,
                     protocol=lambda: {'tasks': [spec]}, task_spec=project.spec, sha=common.sha)
    science.extract(project.path(R13 + '/runtime.py'), {'seed', 'env', 'target_data', 'model_config'}, namespace)
    science.extract(project.path(R14 + '/train.py'), {'evaluate'}, namespace)
    return namespace


def make_t(project, functions, science, seed, device):
    import torch
    path = project.path('source_snapshot/algo/offline_offline/iql_dyn.py')
    sys.path.insert(0, str(path.parents[1]))
    from offline_offline import iql_dyn
    assert Path(iql_dyn.__file__).resolve() == path
    functions['IQL_DYN'] = iql_dyn.IQL_DYN
    functions['__name__'] = '_v12_stage3_R13_T'
    science.extract(project.path(R13 + '/learner.py'), {'Learner', 'Sampler', 'initialize'}, functions)
    config = functions['model_config'](10., STEPS['T'], seed)
    model = functions['initialize'](config, device)
    sampler = functions['Sampler'](functions['target_data'](), seed, 'target_sample', device)
    return model, sampler, config


def make_root(job, project, trial, science, functions, device, writer):
    import gym
    import h5py
    import numpy as np
    import torch
    source, target, _, inputs = load_root_data(project, trial)
    assert inputs == read(OUT / 'ROOT_INPUTS.json')
    receipt = read(OUT / 'ROOT_TRAINING_WEIGHTS.json')
    path = Path(receipt['path'])
    assert sha(path) == receipt['sha256'] and receipt['source_rows_sha256'] == inputs['source_rows_sha256']
    assert receipt['formal_training_seeds'] == list(SEEDS)
    cfg = dict(read(STAGE2 / 'config.json')['learner'])
    # This normal Gym environment supplies the same source-domain dimensions and bounds as the upstream registration.
    source_env = gym.make('Walker2d-v2')
    try:
        cfg.update(state_dim=int(source_env.observation_space.shape[0]), action_dim=int(source_env.action_space.shape[0]),
                   max_action=float(source_env.action_space.high[0]))
        assert (cfg['state_dim'], cfg['action_dim'], cfg['max_action']) == (17, 6, 1.)
        assert np.all(source_env.action_space.high == 1) and np.all(source_env.action_space.low == -1)
    finally:
        source_env.close()
    cfg['metric'] = 'euclidean'
    assert cfg['max_step'] == 1000000 and cfg['batch_size'] == 128 and cfg['temp'] == 3.
    assert cfg['noreg'] is False and cfg['proportion'] == .25 and cfg['reg_weight'] == .5
    namespace = dict(np=np, torch=torch, os=os, h5py=h5py, __name__='_v12_root_buffers')
    science.extract(trial.UPSTREAM / 'algo/utils.py', {'ReplayBuffer', 'OTReplayBuffer', 'load_robust_weights'}, namespace)
    raw_weights = namespace['load_robust_weights'](str(path))
    assert raw_weights.shape == (len(source['observations']),) and np.isfinite(raw_weights).all()
    src = namespace['OTReplayBuffer'](17, 6, device)
    dst = namespace['ReplayBuffer'](17, 6, device)
    src.convert_D4RL(source)
    dst.convert_D4RL(target)
    src.set_weights(raw_weights / (raw_weights.mean() + 1e-12))
    env = functions['env']()
    assert env.action_space.shape[0] == cfg['action_dim'] and env.unwrapped.frame_skip == 4
    seed = job['seed']
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    root = trial.load_upstream('algo/ROOT.py', '_v12_stage3_upstream_ROOT')
    model = root.ROOT(cfg, device)
    return model, src, dst, env, cfg, receipt


def wait_release(job):
    release = Path(job['release'])
    while not release.exists():
        assert os.getppid() == job['controller_pid'], 'Stage 3 controller exited before GPU release'
        time.sleep(.2)
    grant = read(release)
    assert grant['GPU_UUID'] == job['gpu_uuid'] and job['id'] in grant['jobs']
    assert grant['controller_pid'] == os.getppid() and 1 <= len(grant['jobs']) <= CAP


def train(job, legacy):
    bc, common, _, _, trial, science = legacy
    assert os.environ['CUDA_VISIBLE_DEVICES'] == job['gpu_uuid']
    wait_release(job)
    import numpy as np
    import torch
    from tensorboardX import SummaryWriter
    bc.configure_torch(torch)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    group, seed = job['group'], job['seed']
    out = Path(job['output'])
    assert seed in SEEDS and job['steps'] == STEPS[group]
    project = science.Project(job['project'])
    functions = runtime(project, science, common)
    common.private_binding(out)
    writer = SummaryWriter(str(out / 'tb'))
    device = 'cuda:0'
    start = time.monotonic()
    env = None
    sampler = None
    vae_seconds = 0.
    try:
        if group == 'ROOT':
            model, source, target, env, config, weights = make_root(job, project, trial, science, functions, device, writer)
            act = SimpleNamespace(deterministic_action=model.select_action)
            learner_state = lambda: trial.state_snapshot(model)
            update = lambda: model.train(source, target, 128, writer)
        elif group == 'T':
            model, sampler, config = make_t(project, functions, science, seed, device)
            act = model
            learner_state = model.checkpoint
            update = lambda: model.step(sampler.sample(), None)
        else:
            assert group == 'BC-all'
            rows = list(range(5000))
            models = science.bc_definitions(project, rows)
            arrays = functions['target_data']()
            model = models['Actor'](job, device)
            sampler = models['Sampler'](arrays, job, device)
            act = model
            config = dict(beta=0, lr=.0003, batch_size=64, cosine_T_max=200000, hidden_sizes=256,
                          loss='mean action MSE then batch mean', source_rows=0)
            learner_state = model.checkpoint
            update = lambda: model.step(*sampler.sample())
        if env is None:
            env = functions['env']()
        assert env.unwrapped.frame_skip == 4 and env._max_episode_steps == 1000
        common.write(out / 'config.json', dict(job=job, learner=config, source_action_dim_from_env=group == 'ROOT',
                     ROOT_source_next=source_mode() if group == 'ROOT' else None,
                     weight_sha256=weights['sha256'] if group == 'ROOT' else None,
                     target_data_sha256=project.manifest[project.tasks[TASK]['target_dataset']]['sha256'],
                     target_rows=5000, explicit_next=True, evaluation_points=points(group), threads=1,
                     eval_seed_formula='500000+10*(training_seed-501)+episode', deterministic_algorithms=True, TF32=True))
        if group == 'ROOT':
            before = time.monotonic()
            model.train_vae(target, 128, writer)
            torch.cuda.synchronize()
            vae_seconds = time.monotonic() - before
            bc.parent_check(job)
            assert bc.finite_state(torch, model.vae_policy.state_dict())
            common.write(out / 'VAE_COMPLETED.json', dict(updates=10000, seconds=vae_seconds))
        eval_job = dict(job, seed=201 + seed - 501)
        block = time.monotonic()
        seconds = 0.
        evaluations = []
        for step in range(1, STEPS[group] + 1):
            value = update()
            if step % 1000 == 0:
                bc.parent_check(job)
                if group == 'T':
                    for name, loss in zip(('V', 'Q', 'actor'), value):
                        assert math.isfinite(float(loss))
                        writer.add_scalar('loss/' + name, float(loss), step)
                elif group == 'BC-all':
                    assert math.isfinite(float(value))
                    writer.add_scalar('loss/actor', float(value), step)
            if step % EVAL_EVERY[group] == 0:
                torch.cuda.synchronize()
                seconds += time.monotonic() - block
                assert bc.finite_state(torch, learner_state()), f'nonfinite state at {step}'
                timer = time.monotonic()
                before = bc.rng_digest(torch)
                result = functions['evaluate'](act, env, eval_job)
                after = bc.rng_digest(torch)
                assert before == after
                result.update(step=step, training_RNG_sha256_before=before, training_RNG_sha256_after=after,
                              length_statistics=common.length_stats(result['episode_lengths']),
                              evaluation_seconds=time.monotonic() - timer)
                evaluations.append(result)
                common.write(out / f'eval_{step:07d}.json', result)
                writer.add_scalar('evaluation/target_score', result['mean'], step)
                writer.flush()
                print(job['id'], step, '/', STEPS[group], flush=True)
                block = time.monotonic()
        assert model.total_it == STEPS[group]
        checkpoint = out / 'final_checkpoint.pt'
        torch.save(learner_state(), str(checkpoint))
        common.write(out / 'COMPLETED.json', dict(
            job=job, main_steps=model.total_it, VAE_updates=10000 if group == 'ROOT' else 0,
            main_Adam_updates=model.total_it * (1 if group == 'BC-all' else 3), checkpoint_sha256=sha(checkpoint),
            evaluation_episodes=10 * len(evaluations), evaluation_steps=sum(e['steps'] for e in evaluations),
            training_seconds=seconds, vae_seconds=vae_seconds, wall_seconds=time.monotonic() - start,
            eval_seconds=sum(e['evaluation_seconds'] for e in evaluations),
            sampling=None if sampler is None else dict(draws=sampler.draws, stream_sha256=sampler.hash.hexdigest()),
            GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(), ended_utc=common.utc()))
    finally:
        writer.close()
        if env is not None:
            env.close()


def read_eval_records(job, project, common):
    import numpy as np
    out = Path(job['output'])
    spec = project.spec(TASK)
    origin = 500000 + 10 * (job['seed'] - 501)
    events = []
    for step in points(job['group']):
        event = read(out / f'eval_{step:07d}.json')
        assert event['step'] == step and len(event['episode_lengths']) == len(event['raw_returns']) == 10
        assert event['steps'] == sum(event['episode_lengths'])
        assert event['env_seeds'] == list(range(origin, origin + 10))
        assert event['training_RNG_sha256_before'] == event['training_RNG_sha256_after']
        assert event['xml_sha256'] == spec['xml_sha256'] and abs(event['timestep'] - .002) < 1e-12
        assert event['length_statistics'] == common.length_stats(event['episode_lengths'])
        score = (np.asarray(event['raw_returns']) - spec['score_min']) / (spec['score_max'] - spec['score_min']) * 100
        assert np.isfinite(score).all() and np.array_equal(score, event['scores']) and float(score.mean()) == event['mean']
        events.append(event)
    return events


def audit(job, legacy):
    bc, common, _, _, trial, science = legacy
    import numpy as np
    import torch
    bc.configure_torch(torch)
    group, steps = job['group'], STEPS[job['group']]
    out = Path(job['output'])
    done = read(out / 'COMPLETED.json')
    assert done['job'] == job and done['main_steps'] == steps
    checkpoint = out / 'final_checkpoint.pt'
    assert sha(checkpoint) == done['checkpoint_sha256']
    ck = torch.load(str(checkpoint), map_location='cpu')
    assert bc.finite_state(torch, ck)
    if group == 'ROOT':
        assert ck['main_steps'] == steps and done['VAE_updates'] == 10000
        weight_info = read(OUT / 'ROOT_TRAINING_WEIGHTS.json')
        assert read(out / 'config.json')['weight_sha256'] == weight_info['sha256']
        assert sha(weight_info['path']) == weight_info['sha256']
        assert set(ck['networks']) == {'policy', 'v_func', 'q_funcs', 'target_q_funcs', 'vae_policy'}
        optimizers, scheduler = ck['optimizers'], ck['scheduler']
        assert set(optimizers) == {'policy_optimizer', 'v_optimizer', 'q_optimizer', 'vae_policy_optimizer'}
    else:
        assert ck['step'] == steps and done['VAE_updates'] == 0
        scheduler = ck['actor_scheduler']
        names = ('actor_optimizer',) if group == 'BC-all' else ('actor_optimizer', 'V_optimizer', 'Q_optimizer')
        optimizers = {name: ck[name] for name in names}
        if group == 'T':
            assert all(name in ck for name in ('policy', 'Q', 'V', 'target_Q'))
        else:
            assert set(ck) == {'policy', 'actor_optimizer', 'actor_scheduler', 'step', 'job'}
        namespace = '20261016' if group == 'T' else 'V9-stage3-20261031'
        seed_bytes = hashlib.sha256(f"{namespace}|{TASK}|{job['seed']}|target_sample".encode()).digest()[:4]
        rng = np.random.RandomState(int.from_bytes(seed_bytes, 'little'))
        stream = hashlib.sha256()
        for start in range(0, steps, 4096):
            indices = rng.randint(5000, size=64 * min(4096, steps - start)).astype('<i8')
            stream.update(indices.tobytes())
        assert done['sampling'] == dict(draws=steps * 64, stream_sha256=stream.hexdigest())
    for name, optimizer in optimizers.items():
        wanted = 10000 if name == 'vae_policy_optimizer' else steps
        assert optimizer['state'] and all(int(value['step']) == wanted for value in optimizer['state'].values())
    assert scheduler['T_max'] == scheduler['last_epoch'] == steps
    assert done['main_Adam_updates'] == steps * (1 if group == 'BC-all' else 3)
    events = read_eval_records(job, science.Project(job['project']), common)
    assert done['evaluation_episodes'] == len(events) * 10 and done['evaluation_steps'] == sum(e['steps'] for e in events)
    assert not torch.cuda.is_initialized()
    common.write(out / 'AUDIT.json', dict(passed=True, group=group, seed=job['seed'], steps=steps,
                 checkpoint_sha256=sha(checkpoint), completed_sha256=sha(out / 'COMPLETED.json'),
                 raw_evaluation_points=len(events), evaluation_sha256={str(e['step']): sha(out / f"eval_{e['step']:07d}.json") for e in events},
                 CPU_only=True, CUDA_initialized=False, time=common.utc()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['probe', 'weights', 'train', 'audit'])
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    verify_new()
    legacy = load_legacy()
    job = read(args.job)
    auth = read(OUT / 'AUTHORIZATION.json')
    assert auth['scope'] == 'V12_STAGE3_3_GPU_PROCESSES_ONLY' and auth['jobs'][job['id']] == sha(args.job)
    assert job['task'] == TASK and job['gpu_uuid'] == auth['GPU_UUID'] and job['gpu_uuid'] in GPU_UUIDS.values()
    if args.mode == 'probe':
        assert job['id'] == 'probe' and os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
        probe(job, legacy)
    elif args.mode == 'weights':
        assert job['id'] == 'weights' and source_mode() == 'upstream_adjacent'
        assert 'seed' not in job
        wait_release(job)
        trial = legacy[4]
        proof = solver_seed_proof(trial)
        assert read(OUT / 'ROOT_OT_SEED_INDEPENDENCE.json') == proof
        trial.CLI = proof['effective_solver_arguments']
        loader = trial.load_pair
        trial.load_pair = lambda project: load_root_data(project, SimpleNamespace(load_pair=loader))
        trial.weights(job)
    else:
        assert job['group'] in STEPS and job['seed'] in SEEDS and job['id'] == f"{job['group']}_s{job['seed']}"
        dict(train=train, audit=audit)[args.mode](job, legacy)


if __name__ == '__main__':
    main()
