from pathlib import Path
from statistics import mean, stdev
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path('runs_hopper_bellman_b020_5seeds/b0p20/IQL_BELLMAN_FB/hopper-kinematic-legjnt-srcdatatype-medium-tardatatype-medium-medium')
rows=[]
for seed in range(5):
    tb = root / f'r{seed}' / 'tb'
    files = sorted(tb.glob('events.out.tfevents*'))
    acc = EventAccumulator(str(files[-1])); acc.Reload()
    tags = acc.Tags()['scalars']
    score_tag = 'test/target_normalized_score'
    if score_tag not in tags:
        score_tag = [t for t in tags if 'target' in t.lower() and 'normalized' in t.lower() and 'score' in t.lower()][0]
    vals = [x.value for x in acc.Scalars(score_tag)]
    d = {'seed':seed,'final':vals[-1],'best':max(vals),'last5':sum(vals[-5:])/5}
    for tag in ['train/source_weight_mean','train/source_weight_ess_ratio','train/source_weight_at_min_ratio','train/bellman_mismatch_mean','train/corr_bellman_mismatch_source_weight']:
        if tag in tags:
            d[tag.split('/')[-1]] = acc.Scalars(tag)[-1].value
    rows.append(d)

print('seed,final,best,last5,source_weight_mean,source_weight_ess_ratio,source_weight_at_min_ratio,bellman_mismatch_mean,corr_bellman_mismatch_source_weight')
for r in rows:
    print(','.join([str(r['seed']), f"{r['final']:.4f}", f"{r['best']:.4f}", f"{r['last5']:.4f}", f"{r.get('source_weight_mean', float('nan')):.4f}", f"{r.get('source_weight_ess_ratio', float('nan')):.4f}", f"{r.get('source_weight_at_min_ratio', float('nan')):.4f}", f"{r.get('bellman_mismatch_mean', float('nan')):.4f}", f"{r.get('corr_bellman_mismatch_source_weight', float('nan')):.4f}"]))
print('\nsummary')
for k in ['final','best','last5','source_weight_mean','source_weight_ess_ratio','source_weight_at_min_ratio','bellman_mismatch_mean','corr_bellman_mismatch_source_weight']:
    xs=[r[k] for r in rows if k in r]
    print(f'{k},{mean(xs):.4f},{stdev(xs):.4f}')
