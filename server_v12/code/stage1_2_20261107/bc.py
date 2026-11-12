"""Fixed V12 BC runs and read-only verification of their recorded outputs."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import pickle
import random
import time

from common import length_stats, private_binding, read, sha, utc, verify_package, write
from science import Project, bc_definitions, rows_digest, transitions, validate_dimensions


STEPS = 200000
EVAL_POINTS = list(range(5000, STEPS + 1, 5000))


def materialize(job):
    import numpy as np
    project = Project(job['project'])
    task = job['task']
    source = job['stage'] == '1A'
    spec = project.spec(task)
    path = project.path(spec['source_dataset' if source else 'target_dataset'])
    arrays, canonical = transitions(path, source=source, actor_only=True)
    validate_dimensions(arrays, spec)
    if source:
        assert task == 'hopper-morph' and len(canonical) == spec['source_rows']
        rows = np.arange(len(canonical), dtype=np.int64)
    else:
        assert len(canonical) == 5000
        subsets = read(Path(job['base']) / 'TRAJECTORY_SUBSETS.json')
        rows = np.asarray(subsets[task]['subsets'][str(job['trajectories'])]['rows'], dtype=np.int64)
    return project, arrays, rows, rows_digest(canonical[rows]), path


def rng_digest(torch):
    import numpy as np
    values = [random.getstate(), np.random.get_state(), torch.get_rng_state().numpy().tobytes()]
    values.extend(state.cpu().numpy().tobytes() for state in torch.cuda.get_rng_state_all())
    return hashlib.sha256(pickle.dumps(values, protocol=4)).hexdigest()


def configure_torch(torch):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def parent_check(job):
    assert os.getppid() == job['controller_pid'], 'V12 controller exited'


def train(job):
    assert job['seed'] in range(401, 406) and job['stage'] in ('1A', '1B')
    assert job['stage'] == '1A' or job['trajectories'] in (2, 4, 7)
    assert os.environ['CUDA_VISIBLE_DEVICES'] == job['gpu_uuid'] and ',' not in job['gpu_uuid']
    parent_check(job)
    import torch
    from tensorboardX import SummaryWriter
    configure_torch(torch)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    out = Path(job['output'])
    project, arrays, rows, raw_row_hash, path = materialize(job)
    functions = bc_definitions(project, rows)
    actor = functions['Actor'](job, 'cuda:0')
    sampler = functions['Sampler'](tuple(arrays.values()), job, 'cuda:0')
    write(out / 'config.json', dict(
        job=job, steps=STEPS, cosine_T_max=STEPS, lr=3e-4, batch=64, hidden_sizes=256,
        beta=0, loss='action dimension mean squared error, then batch mean', source_in_critic=False,
        observation_scaling='raw', normalization=project.spec(job['task']), evaluation_points=EVAL_POINTS,
        episodes_per_point=10, eval_seed_origin=301, threads=1, selected_rows=len(rows),
        selected_raw_rows_sha256=raw_row_hash, data_path=str(path), data_sha256=sha(path),
        deterministic_algorithms=True, TF32=True, last5_points=EVAL_POINTS[-5:],
    ))
    write(out / 'INITIALIZATION.json', dict(parameter_sha256=actor.digest()))
    private_binding(out)
    env = functions['env'](job['task'])
    assert env.unwrapped.frame_skip == 4 and env._max_episode_steps == 1000
    writer = SummaryWriter(str(out / 'tb'))
    events, training_seconds = [], 0.0
    train_start = time.monotonic()
    block_start = train_start
    try:
        for step in range(1, STEPS + 1):
            loss = actor.step(*sampler.sample())
            if step % 1000 == 0:
                parent_check(job)
                value = float(loss)
                assert math.isfinite(value) and actor.finite(), f'nonfinite BC step {step}'
                writer.add_scalar('loss/actor', value, step)
                write(out / f'progress_{step:06d}.json', dict(step=step, elapsed=time.monotonic() - train_start,
                                                             loss=value, draws=sampler.draws))
            if step % 5000 == 0:
                torch.cuda.synchronize()
                training_seconds += time.monotonic() - block_start
                started = time.monotonic()
                before = rng_digest(torch)
                result = functions['evaluate'](actor, env, job)
                after = rng_digest(torch)
                assert before == after
                result.update(step=step, training_RNG_sha256_before=before, training_RNG_sha256_after=after,
                              length_statistics=length_stats(result['episode_lengths']),
                              evaluation_seconds=time.monotonic() - started)
                events.append(result)
                write(out / f'eval_{step:06d}.json', result)
                writer.add_scalar('evaluation/target_score', result['mean'], step)
                writer.flush()
                print(job['id'], step, '/', STEPS, flush=True)
                block_start = time.monotonic()
        end = time.monotonic()
        checkpoint = out / 'final_checkpoint.pt'
        torch.save(actor.checkpoint(), str(checkpoint))
        write(out / 'COMPLETED.json', dict(
            job=job, steps=actor.total_it, Adam_updates=actor.total_it, finite=actor.finite(),
            training_seconds=training_seconds, wall_seconds=end - train_start, end_monotonic=end,
            evaluation_seconds=sum(e['evaluation_seconds'] for e in events),
            evaluation_episodes=sum(len(e['episode_lengths']) for e in events),
            evaluation_environment_steps=sum(e['steps'] for e in events),
            checkpoint_sha256=sha(checkpoint), model_digest=actor.digest(),
            target_or_source_stream_sha256=sampler.hash.hexdigest(), draws=sampler.draws,
            GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(), ended_utc=utc(),
        ))
    finally:
        writer.close()
        env.close()


def finite_state(torch, value):
    if isinstance(value, torch.Tensor):
        return bool(torch.isfinite(value).all())
    if isinstance(value, dict):
        return all(finite_state(torch, x) for x in value.values())
    if isinstance(value, (list, tuple)):
        return all(finite_state(torch, x) for x in value)
    return not isinstance(value, float) or math.isfinite(value)


def eval_records(out, job, spec, points=EVAL_POINTS, seed_origin=301):
    import numpy as np
    records = [read(out / f'eval_{point:06d}.json') for point in points]
    for point, event in zip(points, records):
        assert event['step'] == point and len(event['episode_lengths']) == len(event['raw_returns']) == 10
        assert event['steps'] == sum(event['episode_lengths'])
        assert event['xml_sha256'] == spec['xml_sha256'] and abs(event['timestep'] - .002) < 1e-12
        assert event['env_seeds'] == list(range(500000 + 10 * (job['seed'] - seed_origin),
                                                500010 + 10 * (job['seed'] - seed_origin)))
        assert event['training_RNG_sha256_before'] == event['training_RNG_sha256_after']
        assert length_stats(event['episode_lengths']) == event['length_statistics']
        scores = (np.asarray(event['raw_returns']) - spec['score_min']) / (spec['score_max'] - spec['score_min']) * 100.
        assert np.isfinite(scores).all() and np.array_equal(scores, event['scores'])
        assert float(scores.mean()) == event['mean']
    return records


def audit(job):
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    import numpy as np
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    out = Path(job['output'])
    completed = read(out / 'COMPLETED.json')
    config = read(out / 'config.json')
    assert completed['job'] == job and completed['steps'] == completed['Adam_updates'] == STEPS
    assert completed['finite'] and completed['draws'] == STEPS * 64
    checkpoint = out / 'final_checkpoint.pt'
    assert sha(checkpoint) == completed['checkpoint_sha256']
    ck = torch.load(str(checkpoint), map_location='cpu')
    assert set(ck) == {'policy', 'actor_optimizer', 'actor_scheduler', 'step', 'job'}
    assert ck['job'] == job and ck['step'] == STEPS and finite_state(torch, ck)
    assert ck['actor_scheduler']['T_max'] == ck['actor_scheduler']['last_epoch'] == STEPS
    adam = ck['actor_optimizer']
    assert len(adam['state']) == len(ck['policy']) and all(int(x['step']) == STEPS for x in adam['state'].values())
    assert adam['param_groups'][0]['lr'] == 0.0
    model_hash = hashlib.sha256(b''.join(x.numpy().tobytes() for x in ck['policy'].values())).hexdigest()
    assert model_hash == completed['model_digest']
    project, _, rows, raw_row_hash, _ = materialize(job)
    assert config['selected_raw_rows_sha256'] == raw_row_hash and config['selected_rows'] == len(rows)
    rng_seed = int.from_bytes(hashlib.sha256(
        f"V9-stage3-20261031|{job['task']}|{job['seed']}|target_sample".encode()).digest()[:4], 'little')
    rng = np.random.RandomState(rng_seed)
    stream = hashlib.sha256()
    for offset in range(0, STEPS, 4096):
        count = min(4096, STEPS - offset) * 64
        stream.update(rows[rng.randint(len(rows), size=count)].astype('<i8').tobytes())
    assert stream.hexdigest() == completed['target_or_source_stream_sha256']
    records = eval_records(out, job, project.spec(job['task']))
    assert completed['evaluation_episodes'] == 400
    assert completed['evaluation_environment_steps'] == sum(r['steps'] for r in records)
    assert not torch.cuda.is_initialized()
    write(out / 'AUDIT.json', dict(passed=True, CPU_only=True, complete_sample_stream_verified=True,
                                  raw_evaluation_points=len(records), checkpoint_sha256=sha(checkpoint),
                                  completed_sha256=sha(out / 'COMPLETED.json'), time=utc()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'audit'])
    parser.add_argument('job', type=Path)
    args = parser.parse_args()
    verify_package()
    job = read(args.job)
    auth = read(Path(job['base']) / 'AUTHORIZATION.json')
    assert auth['jobs'][job['id']] == sha(args.job) and auth['stage'] == job['stage']
    if args.mode == 'train':
        train(job)
    else:
        audit(job)


if __name__ == '__main__':
    main()
