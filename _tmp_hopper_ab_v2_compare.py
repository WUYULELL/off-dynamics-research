from pathlib import Path
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

runs = [
 ('IQL-target', Path('runs_cross_env_iql_target/IQL/hopper-kinematic-legjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb')),
 ('IQL-mix', Path('runs_cross_env_iql_mix/IQL/hopper-kinematic-legjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb')),
 ('IQL_CONST-w033', Path('runs_cross_env_iql_const_w033/IQL_UNC/hopper-kinematic-legjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb')),
 ('IQL_BELLMAN_FB-b0.33', Path('runs_cross_env_iql_bellman_fb/IQL_BELLMAN_FB/hopper-kinematic-legjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb')),
 ('IQL_BELLMAN_FB-b0.20', Path('runs_hopper_bellman_b020_5seeds/b0p20/IQL_BELLMAN_FB/hopper-kinematic-legjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb')),
 ('IGDF', Path('runs_cross_env_igdf/IGDF/hopper-kinematic-legjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb')),
 ('IQL_BELLMAN_AB-v1', Path('r_ab_hopper/IQL_BELLMAN_AB/hopper-kinematic-legjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb')),
 ('IQL_BELLMAN_AB-v2', Path('r_ab_hopper_v2/IQL_BELLMAN_AB/hopper-kinematic-legjnt-srcdatatype-medium-tardatatype-medium-medium/r0/tb')),
]

def read(tb):
    files=sorted(tb.glob('events.out.tfevents*'), key=lambda p:p.stat().st_mtime)
    best=None
    for f in files:
        acc=EventAccumulator(str(f)); acc.Reload(); tags=acc.Tags().get('scalars',[])
        matches=[t for t in tags if 'target' in t.lower() and 'normalized' in t.lower() and 'score' in t.lower()]
        if not matches: continue
        tag='test/target_normalized_score' if 'test/target_normalized_score' in tags else matches[0]
        vals=[x.value for x in acc.Scalars(tag)]
        cur=(len(vals), vals[-1], max(vals), sum(vals[-5:])/min(5,len(vals)))
        if best is None or cur[0] > best[0]: best=cur
    return best
print('method,n_eval,final,best,last5')
for name,tb in runs:
    r=read(tb)
    print(name, 'MISSING' if r is None else ','.join([str(r[0]), f'{r[1]:.4f}', f'{r[2]:.4f}', f'{r[3]:.4f}']), sep=',')
