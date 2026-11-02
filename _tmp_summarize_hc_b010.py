from pathlib import Path
from statistics import mean, stdev
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

root = Path('r_hc_b010/IQL_BELLMAN_FB/halfcheetah-kinematic-footjnt-srcdatatype-medium-tardatatype-medium-medium')
rows=[]
for seed in range(5):
    tb = root / f'r{seed}' / 'tb'
    files = sorted(tb.glob('events.out.tfevents*'), key=lambda p: p.stat().st_mtime)
    # Prefer files with a full 50 evaluations; this skips the earlier 1-step smoke test.
    best_result = None
    for f in files:
        acc = EventAccumulator(str(f)); acc.Reload(); tags = acc.Tags().get('scalars', [])
        matches = [t for t in tags if 'target' in t.lower() and 'normalized' in t.lower() and 'score' in t.lower()]
        if not matches:
            continue
        score_tag = 'test/target_normalized_score' if 'test/target_normalized_score' in tags else matches[0]
        vals = [x.value for x in acc.Scalars(score_tag)]
        result = {'seed':seed,'file':f.name,'n_eval':len(vals),'final':vals[-1],'best':max(vals),'last5':sum(vals[-5:])/min(5,len(vals))}
        for tag in ['train/source_weight_mean','train/source_weight_ess_ratio','train/source_weight_at_min_ratio','train/bellman_mismatch_mean','train/corr_bellman_mismatch_source_weight']:
            if tag in tags:
                result[tag.split('/')[-1]] = acc.Scalars(tag)[-1].value
        if best_result is None or result['n_eval'] > best_result['n_eval'] or f.stat().st_mtime > (tb / best_result['file']).stat().st_mtime:
            best_result = result
    rows.append(best_result)

print('seed,n_eval,final,best,last5,source_weight_mean,source_weight_ess_ratio,source_weight_at_min_ratio,bellman_mismatch_mean,corr_bellman_mismatch_source_weight')
for r in rows:
    print(','.join([str(r['seed']), str(r['n_eval']), f"{r['final']:.4f}", f"{r['best']:.4f}", f"{r['last5']:.4f}", f"{r.get('source_weight_mean',float('nan')):.4f}", f"{r.get('source_weight_ess_ratio',float('nan')):.4f}", f"{r.get('source_weight_at_min_ratio',float('nan')):.4f}", f"{r.get('bellman_mismatch_mean',float('nan')):.4f}", f"{r.get('corr_bellman_mismatch_source_weight',float('nan')):.4f}"]))
print('\nsummary')
for k in ['final','best','last5','source_weight_mean','source_weight_ess_ratio','source_weight_at_min_ratio','bellman_mismatch_mean','corr_bellman_mismatch_source_weight']:
    xs=[r[k] for r in rows if r and k in r]
    print(f'{k},{mean(xs):.4f},{stdev(xs):.4f}')
