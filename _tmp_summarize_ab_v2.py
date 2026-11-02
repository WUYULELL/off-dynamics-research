from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def read_run(root):
    root = Path(root)
    files = sorted(root.glob('**/IQL_BELLMAN_AB/**/tb/events.out.tfevents*'), key=lambda p: p.stat().st_mtime)
    if not files:
        return None
    # choose newest full-looking event
    best = None
    for f in files:
        acc = EventAccumulator(str(f)); acc.Reload(); tags = acc.Tags().get('scalars', [])
        matches=[t for t in tags if 'target' in t.lower() and 'normalized' in t.lower() and 'score' in t.lower()]
        if not matches: continue
        score_tag='test/target_normalized_score' if 'test/target_normalized_score' in tags else matches[0]
        vals=[x.value for x in acc.Scalars(score_tag)]
        cur={'file':f,'acc':acc,'tags':tags,'n_eval':len(vals),'final':vals[-1],'best':max(vals),'last5':sum(vals[-5:])/min(5,len(vals))}
        if best is None or cur['n_eval'] > best['n_eval'] or (cur['n_eval']==best['n_eval'] and f.stat().st_mtime > best['file'].stat().st_mtime):
            best=cur
    return best

for name in ['r_ab_hopper', 'r_ab_hopper_v2']:
    r=read_run(name)
    print('\nRUN', name)
    if not r:
        print('missing'); continue
    print('file', r['file'])
    print('n_eval,final,best,last5')
    print(r['n_eval'], f"{r['final']:.4f}", f"{r['best']:.4f}", f"{r['last5']:.4f}")
    acc=r['acc']; tags=r['tags']
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
            xs=acc.Scalars(tag)
            vals=[x.value for x in xs]
            print(tag, 'last', f"{vals[-1]:.4f}", 'mean', f"{sum(vals)/len(vals):.4f}", 'min', f"{min(vals):.4f}", 'max', f"{max(vals):.4f}")
        else:
            print(tag, 'MISSING')
