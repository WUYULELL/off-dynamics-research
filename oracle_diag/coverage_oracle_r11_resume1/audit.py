"""CPU-only independent scalar/model/sampling audit. Does not retrain."""
from runtime import *
import argparse
import numpy as np
import torch
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def audit(path):
    path = Path(path)
    assert OUT in path.resolve().parents
    done, config = read(path / 'completed.json'), read(path / 'config.json')
    job = done['job']
    check_job(job, for_launch=False)
    assert done['steps'] == job['steps'] and done['finite']
    assert sha(done['final_model']) == done['final_model_sha256']
    model = torch.load(done['final_model'], map_location='cpu')
    assert model['step'] == job['steps']
    assert all(torch.isfinite(v).all() for key in ('policy', 'V', 'Q', 'target_Q') for v in model[key].values())
    assert config['actor_source'] is False and config['learner_observations'] == 'raw'
    events = read(path / 'evaluations.json')['events']
    expected = list(range(10000, job['steps'] + 1, 10000)) if job['phase'] != 'smoke' else [5000]
    assert [v['step'] for v in events] == expected
    ea = EventAccumulator(str(Path(shortpath(path)) / 'tb'), size_guidance={'scalars': 0}).Reload()
    scalars = ea.Scalars('evaluation/target_score')
    assert [v.step for v in scalars] == expected
    for tb, ev in zip(scalars, events):
        assert ev['actual_shifted_object'] and ev['preserve_training_RNG']
        scores = (np.array(ev['raw_returns']) - ev['normalization_min']) / (ev['normalization_max'] - ev['normalization_min']) * 100
        assert np.allclose(scores, ev['scores'], rtol=0, atol=1e-10)
        assert abs(float(scores.mean()) - ev['mean']) < 1e-10
        assert abs(tb.value - ev['mean']) < 1e-4
    sample = read(path / 'sampling.json')
    assert sample['target_draws'] == job['steps'] * 64
    assert sample['source_draws'] == job['steps'] * (64 if job['arm'] in ('G','C','R','U') else 0)
    if sample['source_draws']:
        assert read(path / 'source_alignment.json')['source_labels'] == 'oracle only'
    vals = [v['mean'] for v in events[-5:]]
    result = dict(passed=True, job=job, Last5=float(np.mean(vals)), Final=float(events[-1]['mean']),
        last5_steps=[v['step'] for v in events[-5:]], n_checkpoints=len(vals),
        raw_TensorBoard_independently_checked=True, finite_cpu_checked=True,
        final_model_sha256=done['final_model_sha256'], sampling=sample,
        evaluations_sha256=sha(path / 'evaluations.json'), completed_sha256=sha(path / 'completed.json'))
    write(path / 'audit.json', result)
    return result

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', required=True, type=Path)
    args = p.parse_args()
    print(json.dumps(audit(args.out), ensure_ascii=False), flush=True)
