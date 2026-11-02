# -*- coding: utf-8 -*-
"""Continue G205 from100k to500k; only a <=100k prefix is inherited."""
from runtime import *
import argparse
import random
import runpy
import traceback
import numpy as np
import torch
from learner import initialize, Sampler
from resume_state import restore_training, advance_sampler
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
evaluate = runpy.run_path(str(HERE / 'train.py'))['evaluate']

def run(job_path):
    job = read(job_path); check_job(job)
    assert (job['arm'], job['seed']) == ('G', 205)
    accepted = read(OUT / 'cpu_acceptance.json')
    assert accepted['CPU_split_resume_bitwise_identical'] and accepted['CPU_only']
    rec = read(OUT / 'recovery.json')
    assert sha(rec['checkpoint']) == rec['checkpoint_sha256']
    original = Path(rec['original_run'])
    configure_cpu(); assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0) == rec['expected_GPU']
    out = Path(job['out']); fresh(out)
    start = time.time()
    write(out / 'started.json', dict(pid=os.getpid(), time=start, job_path=str(job_path), resume_step=100000))
    try:
        config = read(original / 'config.json')
        assert config['torch'] == torch.__version__ and config['python'] == sys.version
        assert config['learner'] == model_config(10., 500000, 205)
        config.update(job=job, job_sha256=sha(job_path), code_sha256=code_hashes(),
            protocol_sha256=sha(OUT / 'protocol.json'), resume_step=100000,
            resume_checkpoint=rec['checkpoint'], resume_checkpoint_sha256=rec['checkpoint_sha256'],
            original_config_path=str(original / 'config.json'), original_config_sha256=sha(original / 'config.json'))
        write(out / 'config.json', config)
        model = initialize(config['learner'], 'cuda:0')
        assert model.digest() == read(original / 'initialization.json')['sha256']
        write(out / 'initialization.json', read(original / 'initialization.json'))
        restore_training(model, rec['checkpoint'])
        tar = target_data(); raw, oracle, raw_ids, meta = source_data()
        ids = np.load(OUT / 'scores/G_05.npy')
        target = Sampler(tar, 205, 'target_sample', 'cuda:0')
        source = Sampler(oracle, 205, 'source_sample', 'cuda:0', ids)
        advance_sampler(target, 100000); advance_sampler(source, 100000)
        for name, sample in (('target_sample', target), ('source_sample', source)):
            assert sample.hash.hexdigest() == accepted['stream_hashes'][name]['100000']
        alignment = read(original / 'source_alignment.json')
        assert alignment['mask_sha256'] == hashlib.sha256(source.ids.astype('<i8').tobytes()).hexdigest()
        assert alignment['receipt_sha256'] == sha(ROOT / 'oracle_diag/results/v3_20261008/source_oracle1/completed.json')
        write(out / 'source_alignment.json', alignment)
        write(out / 'resume_state.json', dict(step=model.total_it, model_digest=model.digest(),
            checkpoint_sha256=rec['checkpoint_sha256'], optimizer_scheduler_counter_restored=True,
            target_draws=target.draws, source_draws=source.draws,
            target_stream_sha256=target.hash.hexdigest(), source_stream_sha256=source.hash.hexdigest(),
            scheduler=model.policy_lr_schedule.state_dict(), missing_global_RNG_irrelevant_to_deterministic_update=True))
        e = env(); writer = SummaryWriter(str(Path(shortpath(out)) / 'tb'))
        evaluations = [v for v in read(original / 'evaluations.json')['events'] if v['step'] <= 100000]
        assert [v['step'] for v in evaluations] == list(range(10000, 100001, 10000))
        ea = EventAccumulator(str(Path(shortpath(original)) / 'tb'), size_guidance={'scalars': 0}).Reload()
        prefix_count = 0
        for tag in ('loss/V', 'loss/Q', 'loss/actor', 'evaluation/target_score'):
            for ev in ea.Scalars(tag):
                if ev.step <= 100000:
                    writer.add_scalar(tag, ev.value, ev.step, walltime=ev.wall_time); prefix_count += 1
        writer.flush()
        write(out / 'prefix_provenance.json', dict(original_run=str(original), inherited_through_step=100000,
            original_evaluations_sha256=sha(original / 'evaluations.json'), prefix_scalar_count=prefix_count,
            old_steps_after100k_excluded_from_metrics=True,
            original_TensorBoard_files={str(f):sha(f) for f in (original / 'tb').iterdir() if f.is_file()}))
        write(out / 'evaluations.json', dict(events=evaluations))
        overlap = read(OUT / 'OVERLAP_LOSSES.json')['events']; checked = []
        eval_steps = sum(v['steps'] for v in evaluations)
        # Last progress before the100k evaluation excludes that evaluation; duration is provenance only.
        prior_duration = read(original / 'progress.json')['elapsed']
        for step in range(100001, 500001):
            losses = model.step(target.sample(), source.sample())
            assert model.total_it == step
            if step % 1000 == 0:
                assert model.finite()
                for tag, val in zip(('loss/V', 'loss/Q', 'loss/actor'), losses):
                    value = float(val)
                    if str(step) in overlap[tag]:
                        expected = overlap[tag][str(step)]
                        checked.append(dict(tag=tag, step=step, expected=expected, actual=value, exact=value == expected))
                        if value != expected:
                            write(out / 'overlap_failed.json', dict(checks=checked, tolerance=0))
                            raise RuntimeError('GPU overlap differs at %s/%d; stop; no tolerance relaxation' % (tag, step))
                    writer.add_scalar(tag, value, step)
                if step == 109000:
                    assert len(checked) == 24 and all(v['exact'] for v in checked)
                    for name, sample in (('target_sample', target), ('source_sample', source)):
                        assert sample.hash.hexdigest() == accepted['stream_hashes'][name]['109000']
                    write(out / 'OVERLAP_VERIFIED.json', dict(passed=True, checks=checked,
                        exact_float32=True, both_109k_sampling_streams_match=True,
                        continuation_within_original500k_budget=True))
                write(out / 'progress.json', dict(step=step, budget=500000, resume_from=100000,
                    new_updates=step-100000, new_attempt_elapsed=time.time()-start,
                    original_attempt_elapsed=prior_duration, elapsed=time.time()-start,
                    pid=os.getpid(), target_draws=target.draws, source_draws=source.draws,
                    target_stream_sha256=target.hash.hexdigest(), source_stream_sha256=source.hash.hexdigest(),
                    evaluation_steps=eval_steps, finite=True, stage='confirmation', GPU_overlap_verified=step >= 109000), replace=True)
            if step % 10000 == 0:
                result = evaluate(model, e, 'confirmation', 205); result['step'] = step
                evaluations.append(result); eval_steps += result['steps']
                writer.add_scalar('evaluation/target_score', result['mean'], step); writer.flush()
                write(out / 'evaluations.json', dict(events=evaluations), replace=True)
            if step % 100000 == 0:
                checkpoint = model.checkpoint()
                checkpoint['recovery_random_state'] = dict(target_rng=target.rng.get_state(), source_rng=source.rng.get_state(),
                    target_draws=target.draws, source_draws=source.draws,
                    target_stream_sha256=target.hash.hexdigest(), source_stream_sha256=source.hash.hexdigest(),
                    torch_rng=torch.get_rng_state(), torch_cuda_rng=torch.cuda.get_rng_state_all(),
                    numpy_rng=np.random.get_state(), python_rng=random.getstate())
                torch.save(checkpoint, str(out / ('model_%d.pt' % step)))
        writer.close(); e.close()
        assert read(out / 'OVERLAP_VERIFIED.json')['passed']
        final = out / 'model_500000.pt'
        write(out / 'sampling.json', dict(target_first=target.first, source_first=source.first,
            target_draws=target.draws, source_draws=source.draws,
            target_stream_sha256=target.hash.hexdigest(), source_stream_sha256=source.hash.hexdigest()))
        write(out / 'early_losses.json', dict(trace=[], original_first10_in_memory_trace_unavailable=True,
            exact_24_loss_overlap_evidence=str(out / 'OVERLAP_VERIFIED.json')))
        write(out / 'completed.json', dict(job=job, steps=500000, final_model=str(final), final_model_sha256=sha(final),
            finite=model.finite(), model_digest=model.digest(), elapsed=time.time()-start,
            new_attempt_elapsed=time.time()-start, original_attempt_elapsed=prior_duration,
            elapsed_excludes_original_prefix=True, evaluation_steps=eval_steps, ended=time.time(),
            resumed_from=100000, new_updates=400000, source_attempt=str(original), GPU_overlap_exact=True))
    except BaseException:
        write(out / 'failed.json', dict(pid=os.getpid(), time=time.time(), traceback=traceback.format_exc()))
        raise

if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--job', type=Path, required=True)
    run(p.parse_args().job)
