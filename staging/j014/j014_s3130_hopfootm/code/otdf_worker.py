"""One OTDF training run (adaptive-source batch): official OTDF (groups rho0, rho80, IQLCAL) or OTDFTiered
(rho20T, rho80T, rho100T, rho80R).

Driver written after the official train_otdf.py (commit 41e2e62): official config yaml, the update of
state_dim / action_dim / max_action / weight / proportion / noreg / reg_weight / metric as in train_otdf.py,
official seeding (torch, numpy, cuda, random with the run seed; cudnn deterministic), official
OTReplayBuffer / ReplayBuffer / convert_D4RL, official train_vae (10000 steps) unless noreg, then
policy.train(src, tar, 128) for `steps` steps.
Driver differences (they do not change the training computation):
  * data: source via d4rl.qlearning_dataset(None, dataset=dict) (vendored d4rl 1.1), target via the copied
    call_tar_dataset loop, cost read from the official-format hdf5 ('cost') at an explicit path;
  * only the target environment is evaluated (the official script also evaluates the source environment);
    every evaluation saves and restores the python / numpy / torch / CUDA random states, so evaluations
    (select_action calls rsample) do not change the training random stream;
  * evaluation episodes use env.seed(500000 + 10 * (seed - seed_origin) + episode) (same start states as the
    BC runs of j011/j012/j014); the policy acts with the official select_action (mean action);
  * the actor state_dict is saved at the last `save_last` evaluation points (960k ... 1M) with SHA-256;
  * in smoke / throughput mode (record_returns = false) no return is written anywhere.
"""
import hashlib
import json
import math
import os
from pathlib import Path
import random
import sys
import time
import traceback


def main():
    job = json.loads(Path(sys.argv[1]).read_text(encoding='utf-8'))
    out = Path(job['output'])
    code = Path(job['code'])
    sys.path.insert(0, str(code))
    from asrc_common import write_json, scrub
    try:
        result = run(job, out, code)
        write_json(out/'COMPLETED.json', result)
    except BaseException:
        write_json(out/'FAILED.json', dict(time=time.time(), job=job['id'], traceback=scrub(traceback.format_exc())))
        print(scrub(traceback.format_exc()), flush=True)
        sys.exit(1)


class ScalarLog:
    """Stand-in for the tensorboard writer the official train_vae / train expect (values kept in memory)."""

    def __init__(self):
        self.rows = []

    def add_scalar(self, tag, value, step):
        self.rows.append([tag, float(value), int(step)])


def run(job, out, code):
    from asrc_common import (array_digest, read_json, rel, seed_of, sha256_file, verify_official, write_json)
    import asrc_data
    if job.get('device', 'cuda') == 'cuda':
        assert os.environ.get('CUDA_VISIBLE_DEVICES') == str(job['gpu']), 'CUDA_VISIBLE_DEVICES mismatch'
    if job.get('vendor'):
        sys.path.insert(0, job['vendor'])
    sys.path.insert(0, str(code/'otdf_official'))
    os.environ['D4RL_DATASET_DIR'] = str(code/'private_runtime'/'d4rl_datasets')
    import numpy as np
    import torch
    import yaml
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cuda.matmul.allow_tf32 = True     # torch 1.11 defaults (the official code leaves them unset)
    torch.backends.cudnn.allow_tf32 = True
    device = 'cuda' if job.get('device', 'cuda') == 'cuda' else 'cpu'
    if device == 'cuda':
        assert torch.cuda.is_available() and torch.cuda.device_count() == 1, 'expected exactly one visible GPU'

    official = verify_official(code)
    d4rl = asrc_data.import_d4rl()
    import algo.OTDF as otdf_module
    import algo.utils as utils
    assert Path(otdf_module.__file__).resolve() == (code/'otdf_official'/'algo'/'OTDF.py').resolve(), 'algo.OTDF not official copy'
    assert Path(utils.__file__).resolve() == (code/'otdf_official'/'algo'/'utils.py').resolve(), 'algo.utils not official copy'
    from algo.OTDF import OTDF
    from otdf_tiered import OTDFTiered

    sd, ad = job['state_dim'], job['action_dim']
    seed = int(job['seed'])
    record = bool(job.get('record_returns', True))

    # ------------------------------------------------------------------ environment (target only)
    assert sha256_file(job['xml']) == job['xml_sha256'], 'XML hash mismatch'
    if job.get('env_factory'):                                 # local synthetic test only
        sys.path.insert(0, str(Path(job['env_factory']).parent))
        env = __import__(Path(job['env_factory']).stem).make(sd, ad)
    else:
        import gym
        constructor = getattr(__import__(job['env_module'], fromlist=[job['env_class']]), job['env_class'])
        env = gym.wrappers.TimeLimit(constructor(xml_file=job['xml']), max_episode_steps=1000)
        assert env.unwrapped.frame_skip == 4 and abs(float(env.unwrapped.model.opt.timestep)-.002) < 1e-12
        assert env.observation_space.shape == (sd,) and env.action_space.shape == (ad,)
        assert float(env.action_space.high[0]) == 1.0

    # ------------------------------------------------------------------ official config (train_otdf.py)
    assert sha256_file(code/job['otdf_config']) == official[job['otdf_config']]['sha256']
    with open(str(code/job['otdf_config']), 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    config['metric'] = 'cosine'
    config.update({
        'state_dim': sd,
        'action_dim': ad,
        'max_action': 1.0,
        'weight': bool(job['weight']),
        'proportion': float(job['proportion']),
        'noreg': bool(job['noreg']),
        'reg_weight': float(job['reg_weight']),
    })
    assert config['batch_size'] == 128 and config['eval_freq'] == 10000 and config['max_step'] == 1000000

    # ------------------------------------------------------------------ seed all (official order)
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    tiered = job.get('tiers') is not None
    policy = (OTDFTiered if tiered else OTDF)(config, device)

    nets = dict(q=policy.q_funcs, target_q=policy.target_q_funcs, v=policy.v_func, actor=policy.policy, vae=policy.vae_policy)

    def digest():
        h = hashlib.sha256()
        for net in nets.values():
            for value in net.state_dict().values():
                h.update(value.detach().cpu().numpy().tobytes())
        return h.hexdigest()

    def finite():
        return all(bool(torch.isfinite(p).all()) for net in nets.values() for p in net.parameters())

    init_digest = digest()

    # ------------------------------------------------------------------ data
    t0 = time.monotonic()
    if job.get('synthetic_data'):                              # local synthetic test only
        src_dataset, tar_dataset, cost = _synthetic(job, np)
        data_info = dict(synthetic=True)
    else:
        spec = read_json(code/'tasks.json')
        task = next(t for t in spec['tasks'] if t['id'] == job['task'])
        src_dataset, tar_dataset, data_info = asrc_data.load_task_arrays(
            task, spec, d4rl, [Path(p) for p in job.get('preflight_dirs', [])], job.get('private_data'))
        assert sha256_file(job['cost_file']) == job['cost_sha256'], 'cost file hash mismatch'
        cost = asrc_data.read_hdf5_dict(job['cost_file'])['cost']    # as official call_otdf_cost: dataset['cost']
    src_digest = asrc_data.dataset_digest(src_dataset)
    tar_digest = asrc_data.dataset_digest(tar_dataset)
    for name, got, want in (('source', src_digest, job.get('expected_source_digest')),
                            ('target', tar_digest, job.get('expected_target_digest'))):
        if want is not None:
            assert got['features_sas']['sha256'] == want['features_sas']['sha256'] and got['rows'] == want['rows'], \
                '%s [s, a, s\'] array differs from the preflight manifest' % name
    cost = np.asarray(cost)
    assert cost.shape == (src_digest['rows'],) and np.isfinite(cost).all(), 'cost length / finiteness'

    src_replay_buffer = utils.OTReplayBuffer(sd, ad, device)
    tar_replay_buffer = utils.ReplayBuffer(sd, ad, device)
    src_replay_buffer.convert_D4RL(src_dataset)
    tar_replay_buffer.convert_D4RL(tar_dataset)
    src_replay_buffer.cost = cost
    load_seconds = time.monotonic()-t0

    tier_info = None
    if tiered:
        tiers = job['tiers']
        near = np.load(tiers['near_index_file'], allow_pickle=False)
        assert array_digest(near)['sha256'] == tiers['near_index_sha256'], 'near index file changed'
        policy.configure_tiers(src_replay_buffer.size, near, w_target=tiers['w_target'], w_far=tiers['w_far'],
                               w_near=tiers['w_near'], near_batch=tiers['near_batch'],
                               near_seed=seed_of('near-batch', job['task'], seed))
        tier_info = dict(near_rows=int(len(near)), far_rows=int(len(policy.far_index)), w_target=tiers['w_target'],
                         w_far=tiers['w_far'], w_near=tiers['w_near'], near_batch=tiers['near_batch'],
                         layer=tiers.get('layer'))

    write_json(out/'config.json', dict(job=job, otdf_config=config, data=data_info, source_digest=src_digest,
                                       target_digest=tar_digest, cost_rows=int(len(cost)), tiers=tier_info,
                                       initialization_sha256=init_digest, torch=torch.__version__, numpy=np.__version__,
                                       official_files=official, load_seconds=load_seconds))

    log = ScalarLog()
    started = time.monotonic()
    vae_seconds = 0.
    if not config['noreg']:
        policy.train_vae(tar_replay_buffer, config['batch_size'], log)
        vae_seconds = time.monotonic()-started

    # ------------------------------------------------------------------ evaluation (target env only)
    lo, hi = job['score_min'], job['score_max']
    model_dir = Path(job['model_dir']) if job.get('model_dir') else None
    steps, every, save_last = int(job['steps']), int(job['eval_every']), int(job.get('save_last', 5))
    save_points = set(range(every, steps+1, every)[-save_last:]) if model_dir is not None else set()

    def evaluate(step):
        saved = (random.getstate(), np.random.get_state(), torch.get_rng_state(),
                 torch.cuda.get_rng_state_all() if device == 'cuda' else None)
        returns, lengths, seeds = [], [], []
        t = time.monotonic()
        try:
            for episode in range(10):
                value = 500000+10*(seed-int(job['seed_origin']))+episode
                seeds.append(value)
                env.seed(value)
                state, done, ret, length = env.reset(), False, 0., 0
                while not done:
                    state, reward, done, _ = env.step(policy.select_action(np.array(state)))
                    ret += float(reward)
                    length += 1
                returns.append(ret)
                lengths.append(length)
        finally:
            random.setstate(saved[0])
            np.random.set_state(saved[1])
            torch.set_rng_state(saved[2])
            if saved[3] is not None:
                torch.cuda.set_rng_state_all(saved[3])
        seconds = time.monotonic()-t
        assert all(math.isfinite(r) for r in returns), 'non-finite evaluation return'
        if not record:
            return dict(step=step, episodes=10, finite=True, seconds=seconds, env_steps=int(sum(lengths)))
        scores = [(r-lo)/(hi-lo)*100. for r in returns]
        return dict(step=step, raw_returns=returns, scores=scores, mean=sum(scores)/10., episode_lengths=lengths,
                    env_seeds=seeds, steps=int(sum(lengths)), seconds=seconds)

    saved_models = []
    evaluations, eval_seconds, eval_env_steps = 0, 0., 0
    marks = []
    train_started = time.monotonic()
    try:
        for t in range(steps):
            policy.train(src_replay_buffer, tar_replay_buffer, config['batch_size'], log)
            step = t+1
            if step % 1000 == 0 or step == steps:
                ok = finite()
                if tiered and policy.last_losses is not None:
                    ok = ok and all(bool(torch.isfinite(x)) for x in policy.last_losses)
                assert ok, 'non-finite parameters / losses at step %d' % step
                marks.append([step, time.monotonic()-train_started, eval_seconds])
                write_json(out/'progress.json', dict(step=step, total=steps, elapsed=time.monotonic()-started,
                                                     eval_seconds=eval_seconds, time=time.time(), pid=os.getpid()))
            if step % every == 0:
                e = evaluate(step)
                evaluations += 1
                eval_seconds += e['seconds']
                eval_env_steps += e.get('env_steps', e.get('steps', 0))
                write_json(out/('eval_%07d.json' % step), e)
                if step in save_points:
                    model_dir.mkdir(parents=True, exist_ok=True)
                    path = model_dir/('actor_step_%07d.pt' % step)
                    torch.save(policy.policy.state_dict(), str(path))
                    check = torch.load(str(path), map_location='cpu')
                    assert all(bool(torch.isfinite(v).all()) for v in check.values()), 'saved actor not finite'
                    saved_models.append(dict(step=step, path=rel(path), sha256=sha256_file(path),
                                             bytes=path.stat().st_size))
    finally:
        env.close()
    train_seconds = time.monotonic()-train_started-eval_seconds
    speed = None
    if len(marks) >= 2:
        (s0, w0, e0), (s1, w1, e1) = marks[0], marks[-1]
        if (w1-e1)-(w0-e0) > 0:
            speed = (s1-s0)/((w1-e1)-(w0-e0))
    result = dict(job=job['id'], task=job['task'], group=job['group'], seed=seed, steps=steps, finite=finite(),
                  evaluations=evaluations, record_returns=record, saved_models=saved_models,
                  source_digest=src_digest, target_digest=tar_digest, cost_sha256=job.get('cost_sha256'),
                  tiers=tier_info, tier_stats=getattr(policy, 'tier_stats', None),
                  near_weight=policy.near_weight_stats() if tiered else None,
                  initialization_sha256=init_digest, final_sha256=digest(),
                  load_seconds=load_seconds, vae_seconds=vae_seconds, train_seconds=train_seconds,
                  eval_seconds=eval_seconds, eval_env_steps=eval_env_steps, train_steps_per_second=speed,
                  wall_seconds=time.monotonic()-started+load_seconds,
                  gpu_peak_allocated_bytes=int(torch.cuda.max_memory_allocated()) if device == 'cuda' else 0,
                  vae_loss_last=[r for r in log.rows if r[0] == 'train/VAE loss'][-1:] if log.rows else None,
                  ended=time.time())
    return result


def _synthetic(job, np):
    """Small synthetic data set with the shapes of the task (local tests only)."""
    rng = np.random.RandomState(1234)
    sd, ad = job['state_dim'], job['action_dim']
    n, m = job['synthetic_data']['source_rows'], job['synthetic_data']['target_rows']
    def make(k):
        obs = rng.randn(k, sd).astype(np.float32)
        return {'observations': obs, 'actions': np.tanh(rng.randn(k, ad)).astype(np.float32),
                'next_observations': (obs+.1*rng.randn(k, sd)).astype(np.float32),
                'rewards': rng.rand(k).astype(np.float32), 'terminals': rng.rand(k) < .01}
    return make(n), make(m), -rng.rand(n)


if __name__ == '__main__':
    main()
