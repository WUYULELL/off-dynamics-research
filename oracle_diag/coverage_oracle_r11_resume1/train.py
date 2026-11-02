"""A single authorized fresh run; no continuation, no old output writes."""
from runtime import *
import argparse
import random
import traceback
import numpy as np
import torch
from learner import initialize, Sampler
from tensorboardX import SummaryWriter

def evaluate(model, e, phase, run_seed):
    # An explicit guard also protects against future stochastic env/policy changes.
    saved = (random.getstate(), np.random.get_state(), torch.get_rng_state(), torch.cuda.get_rng_state_all())
    base = 300000 if phase in ('reference', 'smoke') else 400000
    index = 0 if phase in ('reference', 'smoke') else (101, 102, 103).index(run_seed)
    returns, lengths, seeds = [], [], []
    try:
        for j in range(10):
            s = base + 10 * index + j
            seeds.append(s)
            e.seed(s)
            state, done, ret, length = e.reset(), False, 0., 0
            while not done:
                state, reward, done, _ = e.step(model.deterministic_action(state))
                ret += float(reward)
                length += 1
            returns.append(ret)
            lengths.append(length)
    finally:
        random.setstate(saved[0])
        np.random.set_state(saved[1])
        torch.set_rng_state(saved[2])
        torch.cuda.set_rng_state_all(saved[3])
    t = protocol()['tasks'][0]
    scores = (np.asarray(returns) - t['score_min']) / (t['score_max'] - t['score_min']) * 100.
    assert e._max_episode_steps == 1000 and type(e.unwrapped).__name__ == 'Walker2dEnv'
    assert abs(float(e.unwrapped.model.opt.timestep) - .002) < 1e-12
    return dict(raw_returns=returns, scores=scores.tolist(), mean=float(scores.mean()),
        episode_lengths=lengths, env_seeds=seeds, steps=sum(lengths),
        actual_env_class=type(e.unwrapped).__module__, xml_sha256=t['xml_sha256'],
        timestep=float(e.unwrapped.model.opt.timestep), actual_shifted_object=True,
        normalization_min=t['score_min'], normalization_max=t['score_max'],
        preserve_training_RNG=True)

def run(job_path):
    job = read(job_path)
    check_job(job)
    configure_cpu()
    assert torch.cuda.is_available(), 'No CPU fallback for authorized GPU training'
    out = Path(job['out'])
    fresh(out)
    write(out / 'started.json', dict(pid=os.getpid(), time=time.time(), job_path=str(job_path)))
    write(out / 'config.json', dict(job=job, learner=model_config(job['temp'], job['steps'], job['seed']),
        protocol_sha256=sha(OUT / 'protocol.json'), job_sha256=sha(job_path),
        code_sha256=code_hashes(), random_seeds={k: seed(k, job['seed']) for k in
        ('initialization', 'target_sample', 'source_sample', 'D_union', 'R_mask')},
        python=sys.version, torch=torch.__version__, gpu=torch.cuda.get_device_name(0),
        learner_observations='raw', reward_scale=1., source_coefficient=1.,
        V_source=job['arm'] in ('G','C','R','U'), Q_source=job['arm'] in ('G','C','R','U'),
        actor_source=False, batch_target=64, batch_source=64 if job['arm'] in ('G','C','R','U') else 0,
        actor_advantage='pre-V-step target advantage', target_Q_update='every gradient update',
        seed_derivation='uint32 little-endian SHA256(root|task|run_seed|purpose)'))
    try:
        tar = target_data()
        if job['arm'] == 'D':
            with np.load(OUT / 'diagnosis/trajectories.npz') as d:
                extra = [d[k] for k in ('state', 'action', 'next_state', 'reward', 'not_done')]
            tar = [np.concatenate((a, b), axis=0) for a, b in zip(tar, extra)]
        model = initialize(model_config(job['temp'], job['steps'], job['seed']), 'cuda:0')
        write(out / 'initialization.json', dict(sha256=model.digest()))
        target = Sampler(tar, job['seed'], 'D_union' if job['arm'] == 'D' else 'target_sample', 'cuda:0')
        source = None
        if job['arm'] in ('G', 'C', 'R', 'U'):
            raw, oracle, raw_ids, meta = source_data()
            ids = None
            if job['arm'] in ('G', 'C'):
                ids = np.load(OUT / ('scores/%s_%02d.npy' % (job['arm'], round(job['ratio'] * 100))))
            if job['arm'] == 'R':
                ids = np.sort(np.random.RandomState(seed('R_mask', job['seed'])).choice(len(raw_ids),
                    size=int(np.floor(job['ratio'] * len(raw_ids))), replace=False))
                np.save(out / 'R_mask.npy', ids)
            source = Sampler(oracle, job['seed'], 'source_sample', 'cuda:0', ids)
            write(out / 'source_alignment.json', dict(n=len(raw_ids), selected=len(source.ids),
                source_labels='oracle only', raw_row_alignment_exact=True,
                receipt=str(ROOT / 'oracle_diag/results/v3_20261008/source_oracle1/completed.json'),
                receipt_sha256=sha(ROOT / 'oracle_diag/results/v3_20261008/source_oracle1/completed.json'),
                mask_sha256=hashlib.sha256(source.ids.astype('<i8').tobytes()).hexdigest()))
        e = env()
        writer = SummaryWriter(str(Path(shortpath(out)) / 'tb'))
        evaluations, trace = [], []
        start = time.time()
        eval_steps = 0
        for step in range(1, job['steps'] + 1):
            losses = model.step(target.sample(), None if source is None else source.sample())
            if step <= 10:
                trace.append(dict(step=step, V=float(losses[0]), Q=float(losses[1]), actor=float(losses[2])))
            if step % 1000 == 0 or step == job['steps']:
                assert model.finite(), 'Non-finite model'
                write(out / 'progress.json', dict(step=step, budget=job['steps'], elapsed=time.time()-start,
                    pid=os.getpid(), target_draws=target.draws, source_draws=0 if source is None else source.draws,
                    target_stream_sha256=target.hash.hexdigest(), source_stream_sha256=None if source is None else source.hash.hexdigest(),
                    evaluation_steps=eval_steps, finite=True, stage=job['phase']), replace=True)
                for k, v in zip(('V', 'Q', 'actor'), losses):
                    writer.add_scalar('loss/' + k, float(v), step)
            if step % 10000 == 0 or step == job['steps']:
                result = evaluate(model, e, job['phase'], job['seed'])
                result['step'] = step
                evaluations.append(result)
                eval_steps += result['steps']
                writer.add_scalar('evaluation/target_score', result['mean'], step)
                writer.flush()
                write(out / 'evaluations.json', dict(events=evaluations), replace=True)
            if step % 100000 == 0 or step == job['steps']:
                torch.save(model.checkpoint(), str(out / ('model_%d.pt' % step)))
        writer.close()
        e.close()
        final_path = out / ('model_%d.pt' % job['steps'])
        write(out / 'sampling.json', dict(target_first=target.first, source_first=None if source is None else source.first,
            target_draws=target.draws, source_draws=0 if source is None else source.draws,
            target_stream_sha256=target.hash.hexdigest(), source_stream_sha256=None if source is None else source.hash.hexdigest()))
        write(out / 'early_losses.json', dict(trace=trace))
        write(out / 'completed.json', dict(job=job, steps=job['steps'], final_model=str(final_path),
            final_model_sha256=sha(final_path), finite=model.finite(), model_digest=model.digest(),
            elapsed=time.time()-start, evaluation_steps=eval_steps, ended=time.time()))
    except BaseException:
        write(out / 'failed.json', dict(pid=os.getpid(), time=time.time(), traceback=traceback.format_exc()))
        raise

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--job', type=Path, required=True)
    run(p.parse_args().job)
