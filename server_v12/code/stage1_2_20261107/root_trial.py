"""V12 stage 2 adapter for the pinned upstream ROOT learner and OT solver."""
import argparse
import ast
import gc
import hashlib
import importlib.util
import math
import os
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace
import xml.etree.ElementTree as ET

from bc import configure_torch, eval_records, finite_state, parent_check, rng_digest
from common import length_stats, private_binding, read, sha, utc, verify_package, write
from science import HERE, Project, TARGET_KEYS, V9, extract, rows_digest, transitions, validate_dimensions


UPSTREAM = HERE / 'upstream_root'
TASK = 'walker2d-morph'
SEED = 901
POINTS = [10000, 20000]
VAE_UPDATES = 10000
CLI = dict(policy='ROOT', env=TASK, srctype='medium', tartype='medium', seed=SEED, steps=20000,
           weight=False, noreg=False, reg_weight=.5, epsilon=.01, metric='euclidean',
           lambda_src=.05, lambda_tar=.5, filter_threshold=0.)


def load_upstream(relative, module_name):
    """Use the pinned file even when another repository exposes a package named algo."""
    path = (UPSTREAM / relative).resolve()
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    assert Path(module.__file__).resolve() == path
    return module


def dataset_inventory(path):
    import h5py
    import numpy as np
    out = {}
    with h5py.File(path, 'r') as handle:
        def add(name, value):
            if isinstance(value, h5py.Dataset):
                array = np.asarray(value[()])
                out[name] = dict(shape=list(array.shape), dtype=array.dtype.str,
                                 array_sha256=hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest())
        handle.visititems(add)
    return out


def upstream_reference_scores():
    tree = ast.parse((UPSTREAM / 'algo/get_normalized_score.py').read_text(encoding='utf-8'))
    node, = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(target, ast.Name) and target.id == 'REF_SCORES' for target in n.targets)]
    return ast.literal_eval(node.value)


def prepare(job):
    import numpy as np
    import yaml
    out = Path(job['output'])
    project = Project(job['project'])
    scores = upstream_reference_scores()
    comparisons = {}
    for task, xml_name in (('walker2d-morph', 'walker'), ('hopper-morph', 'hopper')):
        spec = project.spec(task)
        name = task.replace('-', '_') + '_medium.hdf5'
        local = project.path(spec['target_dataset'])
        upstream = UPSTREAM / 'datasets' / name
        a, b = dataset_inventory(local), dataset_inventory(upstream)
        fields = {key: dict(upstream=b.get(key), transfer=a.get(key), equal=a.get(key) == b.get(key))
                  for key in sorted(set(a) | set(b))}
        arrays, _ = transitions(local)
        explicit = arrays['next_observations'][:-1]
        shifted = arrays['observations'][1:]
        different = (explicit != shifted).any(axis=1)
        terminal = arrays['terminals'][:-1].reshape(-1).astype(bool)
        root_xml = UPSTREAM / 'envs/assets' / f'{xml_name}_morph.xml'
        xml_timestep = float(ET.parse(root_xml).getroot().find('option').attrib['timestep'])
        comparisons[task] = dict(
            arrays=fields, upstream_hdf5_sha256=sha(upstream), transfer_hdf5_sha256=sha(local),
            arrays_equal=a == b, original_ROOT_rows=len(shifted), explicit_next_rows=len(terminal) + 1,
            shifted_next_rows_differ=int(np.count_nonzero(different)),
            shifted_next_nonterminal_rows_differ=int(np.count_nonzero(different & ~terminal)),
            upstream_xml_sha256=sha(root_xml), transfer_xml_sha256=spec['xml_sha256'],
            xml_bytes_equal=sha(root_xml) == spec['xml_sha256'], timestep=xml_timestep,
            environment_class='Walker2dEnv v3' if xml_name == 'walker' else 'HopperEnv v3',
            time_limit=1000, ROOT_score_constants=scores[task.split('-')[0]]['morphology'],
            V12_score_constants=[spec['score_min'], spec['score_max']],
            ROOT_configuration=yaml.safe_load((UPSTREAM / 'config' / (task.replace('-', '_') + '.yaml')).read_text()),
        )
    write(out / 'ROOT_DATA_ENV_COMPARISON.json', comparisons)
    assert all(x['arrays_equal'] and x['xml_bytes_equal'] and x['timestep'] == .002 for x in comparisons.values())
    source, target, ids, inputs = load_pair(project)
    write(out / 'ROOT_INPUTS.json', inputs)
    raw = comparisons[TASK]['ROOT_configuration']
    effective = dict(raw, state_dim=source['observations'].shape[1], action_dim=source['actions'].shape[1],
                     max_action=1.0, **{k: CLI[k] for k in ('weight', 'noreg', 'reg_weight', 'filter_threshold')})
    effective['metric'] = CLI['metric']
    assert effective['action_dim'] == 6 and effective['max_step'] == 1000000 and effective['batch_size'] == 128
    assert effective['eval_freq'] == 10000 and effective['temp'] == 3.0
    assert np.max(np.abs(target['actions'])) <= effective['max_action'] + 1e-6
    write(out / 'config.json', dict(
        stage='2', task=TASK, commit=read(HERE / 'ROOT_PROVENANCE.json')['commit'], seed=SEED,
        original_yaml=raw, CLI_defaults_effective_except_seed_and_short_loop=CLI,
        learner=effective, source_rows=len(ids), target_rows=len(target['observations']),
        batch_sizes=dict(source=128, target=128), main_updates=20000, VAE_pretraining_updates=VAE_UPDATES,
        VAE_lr=.001, VAE_hidden=750, VAE_latent_dim=12, VAE_reg_beta=.4, VAE_reg_samples=10,
        solver=dict(epsilon=.01, lambda_src=.05, lambda_tar=.5, cost='Euclidean',
                    scale_cost='max_cost', source_chunk_size=10000, source_chunks=math.ceil(len(ids) / 10000),
                    Sinkhorn_threshold=1e-5, Sinkhorn_max_iterations=1000, feature_dtype='float16',
                    features='concatenated s, a, explicit next state standardized by combined domain moments',
                    training_source_clip=5., training_normalization='raw weight / (raw mean + 1e-12)'),
        evaluation=dict(points=POINTS, episodes=10, target_only=True, seed_origin=201,
                        metric='mean normalized target score per point', Last5_applicable=False),
        threads=1, gpu_uuid=job['gpu_uuid'], separate_process_for_JAX_and_Torch=True,
        runtime=dict(deterministic_algorithms=True, cudnn_deterministic=True, cudnn_benchmark=False,
                     TF32=True, CUBLAS_WORKSPACE_CONFIG=':4096:8'),
        adaptation='local verified HDF5 and explicit next observations, V12 evaluation and resource logging',
    ))


def load_pair(project):
    spec = project.spec(TASK)
    source_path = project.path(spec['source_dataset'])
    target_path = project.path(spec['target_dataset'])
    source, ids = transitions(source_path, source=True)
    target, target_ids = transitions(target_path)
    validate_dimensions(source, spec)
    validate_dimensions(target, spec)
    assert len(ids) == spec['source_rows'] and len(target_ids) == spec['target_rows']
    return source, target, ids, dict(source=str(source_path), source_sha256=sha(source_path),
                                      source_rows=len(ids), source_rows_sha256=rows_digest(ids),
                                      target=str(target_path), target_sha256=sha(target_path),
                                      target_rows=len(target_ids), explicit_next=True)


def check_gpu(job):
    parent_check(job)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == job['gpu_uuid'] and ',' not in job['gpu_uuid']


def weights(job):
    check_gpu(job)
    import h5py
    import jax
    import jax.numpy as jnp
    import jaxlib
    import numpy as np
    import ott
    from importlib.metadata import version
    assert jax.__version__ == '0.4.9' and jaxlib.__version__.split('+')[0] == '0.4.9'
    assert version('ott-jax') == '0.4.5' and version('jaxopt') == '0.8.3'
    devices = jax.devices()
    assert jax.default_backend() in ('gpu', 'cuda') and len(devices) == 1 and devices[0].platform in ('gpu', 'cuda')
    out = Path(job['output'])
    project = Project(job['project'])
    source, target, ids, inputs = load_pair(project)
    assert inputs == read(out / 'ROOT_INPUTS.json')
    functions = dict(np=np, jax=jax, jnp=jnp, ott=ott, os=os, gc=gc)
    extract(UPSTREAM / 'robust_ot_solver.py', {'solve_robust_ot', 'filter_dataset', 'compute_and_save_weights'}, functions)

    def view(dataset):
        return SimpleNamespace(state=dataset['observations'], action=dataset['actions'], next_state=dataset['next_observations'])

    path = out / 'robust_weights.hdf5'
    assert not path.exists()
    start = time.monotonic()
    result = functions['compute_and_save_weights'](view(source), view(target), str(path), SimpleNamespace(**CLI))
    elapsed = time.monotonic() - start
    with h5py.File(path, 'r') as f:
        assert np.array_equal(f['cost'][()], result)
    assert result.shape == (len(ids),) and np.isfinite(result).all() and (result >= 0).all() and result.mean() > 0
    normalized = (result / (result.mean() + 1e-12)).astype(np.float32)
    assert np.isfinite(normalized).all()

    def stats(values):
        q = np.quantile(values, [0, .01, .25, .5, .75, .95, .99, 1])
        return dict(mean=float(values.mean()), quantiles=dict(zip(('min', 'p01', 'p25', 'p50', 'p75', 'p95', 'p99', 'max'), q.tolist())))

    write(out / 'WEIGHTS_COMPLETED.json', dict(
        completed=True, seconds=elapsed, path=str(path), sha256=sha(path), source_rows_sha256=rows_digest(ids),
        rows=len(ids), source_chunks=math.ceil(len(ids) / 10000), raw=stats(result), normalized_float32=stats(normalized),
        fraction_zero=float(np.mean(result == 0)), fraction_clipped_at_5=float(np.mean(normalized > 5)),
        fraction_at_cap_after_clip=float(np.mean(normalized >= 5)), clipped=stats(np.minimum(normalized, 5)),
        backend='gpu', jax_device=str(devices[0]), CUDA_VISIBLE_DEVICES=os.environ['CUDA_VISIBLE_DEVICES'],
        jax=version('jax'), jaxlib=version('jaxlib'), ott_jax=version('ott-jax'), jaxopt=version('jaxopt'), time=utc(),
    ))


def state_snapshot(policy):
    networks = ('policy', 'v_func', 'q_funcs', 'target_q_funcs', 'vae_policy')
    optimizers = ('policy_optimizer', 'v_optimizer', 'q_optimizer', 'vae_policy_optimizer')
    return dict(networks={key: getattr(policy, key).state_dict() for key in networks},
                optimizers={key: getattr(policy, key).state_dict() for key in optimizers},
                scheduler=policy.policy_lr_schedule.state_dict(), main_steps=policy.total_it)


def train(job):
    check_gpu(job)
    import numpy as np
    import torch
    from tensorboardX import SummaryWriter
    configure_torch(torch)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    out = Path(job['output'])
    cfg = read(out / 'config.json')
    assert cfg['seed'] == SEED and cfg['main_updates'] == 20000 and cfg['stage'] == '2'
    assert cfg['learner']['max_step'] == 1000000
    project = Project(job['project'])
    source, target, _, inputs = load_pair(project)
    assert inputs == read(out / 'ROOT_INPUTS.json')
    weight_receipt = read(out / 'WEIGHTS_COMPLETED.json')
    path = out / 'robust_weights.hdf5'
    assert sha(path) == weight_receipt['sha256'] and weight_receipt['source_rows_sha256'] == inputs['source_rows_sha256']
    import h5py
    buffers = dict(np=np, torch=torch, os=os, h5py=h5py)
    extract(UPSTREAM / 'algo/utils.py', {'ReplayBuffer', 'OTReplayBuffer', 'load_robust_weights'}, buffers)
    robust = buffers['load_robust_weights'](str(path))
    assert robust.shape == (inputs['source_rows'],) and np.isfinite(robust).all()
    src = buffers['OTReplayBuffer'](17, 6, 'cuda:0')
    tar = buffers['ReplayBuffer'](17, 6, 'cuda:0')
    src.convert_D4RL(source)
    tar.convert_D4RL(target)
    src.set_weights(robust / (robust.mean() + 1e-12))
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    root_module = load_upstream('algo/ROOT.py', '_v12_root_upstream_model')
    policy = root_module.ROOT(cfg['learner'], 'cuda:0')
    private_binding(out)
    environment = dict(task_spec=project.spec, sha=sha, np=np, torch=torch, random=random)
    extract(project.path(V9 + '/runtime.py'), {'env'}, environment)
    extract(project.path('source_snapshot/oracle_diag/coverage_raw_r14/train.py'), {'evaluate'}, environment)
    env = environment['env'](TASK)
    assert env.unwrapped.frame_skip == 4 and env._max_episode_steps == 1000
    scores = load_upstream('algo/get_normalized_score.py', '_v12_root_upstream_scores')
    writer = SummaryWriter(str(out / 'tb'))
    try:
        started = time.monotonic()
        policy.train_vae(tar, 128, writer)
        torch.cuda.synchronize()
        vae_seconds = time.monotonic() - started
        parent_check(job)
        assert finite_state(torch, policy.vae_policy.state_dict())
        write(out / 'VAE_COMPLETED.json', dict(updates=VAE_UPDATES, seconds=vae_seconds))
        total_training = 0.
        eval_times = []
        start_block = time.monotonic()
        for step in range(1, 20001):
            policy.train(src, tar, 128, writer)
            if step % 1000 == 0:
                parent_check(job)
                assert finite_state(torch, state_snapshot(policy)), f'nonfinite ROOT at step {step}'
                write(out / f'progress_{step:06d}.json', dict(step=step, finite=True))
            if step in POINTS:
                torch.cuda.synchronize()
                total_training += time.monotonic() - start_block
                began = time.monotonic()
                before = rng_digest(torch)
                event = environment['evaluate'](SimpleNamespace(deterministic_action=policy.select_action), env, job)
                after = rng_digest(torch)
                assert before == after
                eval_times.append(time.monotonic() - began)
                event.update(step=step, training_RNG_sha256_before=before, training_RNG_sha256_after=after,
                             length_statistics=length_stats(event['episode_lengths']),
                             ROOT_author_normalized_score=scores.get_normalized_score(TASK, float(np.mean(event['raw_returns']))),
                             evaluation_seconds=eval_times[-1])
                write(out / f'eval_{step:06d}.json', event)
                writer.add_scalar('test/target_normalized_score', event['mean'], step)
                writer.add_scalar('test/author_normalized_score_same_episodes', event['ROOT_author_normalized_score'], step)
                writer.flush()
                print('ROOT', step, '/ 20000', flush=True)
                start_block = time.monotonic()
        path = out / 'final_checkpoint.pt'
        torch.save(state_snapshot(policy), str(path))
        write(out / 'TRAINING_COMPLETED.json', dict(
            main_steps=policy.total_it, VAE_updates=VAE_UPDATES, main_Adam_updates=60000,
            training_seconds=total_training, VAE_seconds=vae_seconds, evaluation_seconds=eval_times,
            checkpoint_sha256=sha(path), weight_sha256=weight_receipt['sha256'], GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            estimated_1M_seconds_without_weight_computation=vae_seconds + 50 * total_training + 50 * sum(eval_times), time=utc(),
        ))
    finally:
        writer.close()
        env.close()


def audit(job):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(job['output'])
    complete = read(out / 'TRAINING_COMPLETED.json')
    assert complete['main_steps'] == 20000 and complete['VAE_updates'] == 10000
    checkpoint = out / 'final_checkpoint.pt'
    assert sha(checkpoint) == complete['checkpoint_sha256']
    ck = torch.load(str(checkpoint), map_location='cpu')
    assert finite_state(torch, ck) and ck['main_steps'] == 20000
    assert ck['scheduler']['T_max'] == 1000000 and ck['scheduler']['last_epoch'] == 20000
    for name, opt in ck['optimizers'].items():
        assert opt['state'] and all(int(entry['step']) == (10000 if name == 'vae_policy_optimizer' else 20000)
                                    for entry in opt['state'].values())
    records = eval_records(out, job, Project(job['project']).spec(TASK), POINTS, seed_origin=201)
    author_min, author_max = upstream_reference_scores()['walker2d']['morphology']
    for event in records:
        score = 100. * (np.mean(event['raw_returns']) - author_min) / (author_max - author_min)
        assert math.isclose(score, event['ROOT_author_normalized_score'], rel_tol=1e-12, abs_tol=1e-12)
    weights_sha = sha(out / 'robust_weights.hdf5')
    assert complete['weight_sha256'] == read(out / 'WEIGHTS_COMPLETED.json')['sha256'] == weights_sha
    assert not torch.cuda.is_initialized()
    write(out / 'AUDIT.json', dict(passed=True, main_updates=20000, VAE_updates=10000,
                                  evaluation_episodes=20, checkpoint_sha256=sha(checkpoint),
                                  weight_sha256=weights_sha, CPU_only=True, time=utc()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['prepare', 'weights', 'train', 'audit'])
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    verify_package()
    job = read(args.job)
    auth = read(Path(job['output']) / 'AUTHORIZATION.json')
    assert auth['job_sha256'] == sha(args.job) and job['stage'] == '2'
    assert job['seed'] == SEED and job['task'] == TASK
    dict(prepare=prepare, weights=weights, train=train, audit=audit)[args.mode](job)


if __name__ == '__main__':
    main()
