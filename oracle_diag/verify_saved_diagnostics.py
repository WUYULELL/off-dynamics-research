"""Independent NumPy rank/geometry checks on saved evidence, CPU only."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.stats import rankdata
HERE=Path(__file__).resolve().parent;OUT=HERE/'results/verified_20261003';D=OUT/'diagnostics'


def rank_correlation(a,b):
    x=rankdata(a);y=rankdata(b);x-=x.mean();y-=y.mean()
    den=np.linalg.norm(x)*np.linalg.norm(y)
    return float(x@y/den) if den else None


def main():
    checks=0;max_error=0
    geometry=json.loads((D/'stage1_stage2a.json').read_text())
    for env,report in geometry.items():
        g=np.load(D/(env+'_geometry.npz'));o=np.load(D/(env+'_source_oracle.npz'))
        ids=g['query_indices'];assert len(set(ids.tolist()))==4999;checks+=1
        assert np.array_equal(o['indices'],np.arange(4999));checks+=1
        real=np.sqrt(np.sum(((o['source_next'][ids]-o['oracle_next'][ids])/o['target_state_std'])**2,axis=1))
        max_error=max(max_error,float(np.max(np.abs(real-g['real_mismatch']))));checks+=len(ids)
        for key,name in [('pred_a','A_knn_next_mean'),('pred_b','B_displacement'),('pred_c','C_no_shift')]:
            error=np.sqrt(np.sum(((g[key]-o['oracle_next'][ids])/o['target_state_std'])**2,axis=1))
            assert np.isclose(error.mean(),report['stage2a'][name]['error']['mean'],atol=1e-12);checks+=1
    for path in D.glob('*_bellman.npz'):
        a=np.load(path);r=json.loads(path.with_suffix('.json').read_text())
        for key,method in [('knn','knn'),('diff','diff_mean_state'),('diff_per_neighbor','diff_per_neighbor_sensitivity')]:
            observed=rank_correlation(a[key],a['oracle']);reported=r['metrics'][method]['global_spearman']
            assert np.isclose(observed,reported,atol=1e-12);checks+=1
            values=[rank_correlation(a[key][i:i+128],a['oracle'][i:i+128]) for i in range(0,len(a[key])//128*128,128)]
            assert np.isclose(np.mean([v for v in values if v is not None]),r['metrics'][method]['batch_spearman_mean'],atol=1e-12);checks+=1
    result=dict(passed=True,checks=checks,max_geometry_error=max_error,
        scope='Saved-array alignment, unique query count, independent L2 and rank correlations; no simulator/model rerun claimed')
    (OUT/'independent_checks.json').write_text(json.dumps(result,indent=2),encoding='utf-8');print(result)


if __name__=='__main__':main()
