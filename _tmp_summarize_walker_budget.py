from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

base = Path('runs_walker2d_bellman_budget_sweep')
rows=[]
for b in ['b0p10','b0p20','b0p33','b0p50']:
    tb = base/b/'IQL_BELLMAN_FB'/'walker2d-kinematic-footjnt-srcdatatype-medium-tardatatype-medium-medium'/'r0'/'tb'
    files = sorted(tb.glob('events.out.tfevents*'))
    if not files:
        rows.append((b,'MISSING'))
        continue
    acc=EventAccumulator(str(files[-1])); acc.Reload(); tags=acc.Tags().get('scalars',[])
    score_tag = 'test/target_normalized_score' if 'test/target_normalized_score' in tags else [t for t in tags if 'target' in t.lower() and 'normalized' in t.lower() and 'score' in t.lower()][0]
    vals=[x.value for x in acc.Scalars(score_tag)]
    d={'budget':b,'n_eval':len(vals),'final':vals[-1],'best':max(vals),'last5':sum(vals[-5:])/min(5,len(vals))}
    for tag in ['train/source_weight_mean','train/source_weight_ess_ratio','train/source_weight_at_min_ratio','train/bellman_mismatch_mean','train/corr_bellman_mismatch_source_weight']:
        if tag in tags:
            d[tag.split('/')[-1]]=acc.Scalars(tag)[-1].value
    rows.append(d)
print('budget,n_eval,final,best,last5,source_weight_mean,source_weight_ess_ratio,source_weight_at_min_ratio,bellman_mismatch_mean,corr_bellman_mismatch_source_weight')
for r in rows:
    if isinstance(r, tuple): print(','.join(r)); continue
    print(','.join([r['budget'], str(r['n_eval']), f"{r['final']:.4f}", f"{r['best']:.4f}", f"{r['last5']:.4f}", f"{r.get('source_weight_mean',float('nan')):.4f}", f"{r.get('source_weight_ess_ratio',float('nan')):.4f}", f"{r.get('source_weight_at_min_ratio',float('nan')):.4f}", f"{r.get('bellman_mismatch_mean',float('nan')):.4f}", f"{r.get('corr_bellman_mismatch_source_weight',float('nan')):.4f}"]))
