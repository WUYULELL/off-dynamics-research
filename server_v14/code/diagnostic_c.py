"""V13 cache residuals along original raw source trajectories."""
import numpy as np

from core import csv_write,read,sha,stats,utc,write
from context import RUN


def run(inputs,out):
    steps = inputs.source_steps[inputs.raw_ids]
    index = np.minimum(steps//20,20)
    e0,e1,boundary = (inputs.array(name) for name in ('e0','e1','boundary'))
    threshold = read(RUN/'config.json')['residual_filter']['threshold']
    assert threshold==.0840
    kept = e1<=threshold
    assert np.array_equal(np.flatnonzero(kept),inputs.array('selected_positions'))
    rows = []
    for j in range(21):
        m = index==j
        a,b = stats(e0[m]),stats(e1[m])
        rows.append(dict(step_bin=f'{j*20}-{j*20+19}' if j<20 else '>=400',n=int(m.sum()),
            e0_mean=a['mean'],e0_median=a['median'],e1_mean=b['mean'],e1_median=b['median'],
            corrected_boundary_fraction=float(boundary[m].mean()) if m.any() else None,
            retained_fraction=float(kept[m].mean()) if m.any() else None))
    assert sum(r['n'] for r in rows)==999998
    result = dict(completed=True,bins=rows,threshold=threshold,source_steps_from='Full raw chronological source segments; map cached row_id back to raw within-segment step.',
                  canonical_rows=999998,new_training_updates=0,environment_steps=0,GPU_jobs=0)
    csv_write(out/'C_residual_by_source_step.csv',rows)
    write(out/'RESULT.json',result)
    write(out/'COMPLETED.json',dict(completed=True,result_sha256=sha(out/'RESULT.json'),ended_utc=utc()))
    return result
