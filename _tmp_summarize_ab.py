from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

roots = [Path('r_ab_hopper'), Path('runs')]
files = []
for root in roots:
    if root.exists():
        files += list(root.glob('**/IQL_BELLMAN_AB/**/tb/events.out.tfevents*'))
files = sorted(files, key=lambda p: p.stat().st_mtime)
print('event_files')
for f in files:
    print(f)

if not files:
    raise SystemExit('NO IQL_BELLMAN_AB EVENTS FOUND')

f = files[-1]
acc = EventAccumulator(str(f)); acc.Reload()
tags = acc.Tags().get('scalars', [])
print('\nusing', f)
score_tag = 'test/target_normalized_score' if 'test/target_normalized_score' in tags else [t for t in tags if 'target' in t.lower() and 'normalized' in t.lower() and 'score' in t.lower()][0]
vals = [x.value for x in acc.Scalars(score_tag)]
print('n_eval,final,best,last5')
print(len(vals), vals[-1], max(vals), sum(vals[-5:])/min(5,len(vals)))
print('\nmechanism_last')
for tag in [
 'train/adaptive_source_budget',
 'train/adaptive_source_budget_batch',
 'train/bellman_mismatch_source_mean',
 'train/bellman_mismatch_target_mean',
 'train/bellman_mismatch_ratio',
 'train/source_weight_mean',
 'train/source_weight_min',
 'train/source_weight_max',
 'train/source_weight_ess_ratio',
 'train/source_weight_at_min_ratio',
 'train/corr_bellman_mismatch_source_weight',
]:
    if tag in tags:
        xs = acc.Scalars(tag)
        print(tag, xs[-1].step, xs[-1].value, 'min', min(x.value for x in xs), 'max', max(x.value for x in xs))
    else:
        print(tag, 'MISSING')
