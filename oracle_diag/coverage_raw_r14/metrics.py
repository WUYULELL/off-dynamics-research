"""Precommitted descriptive metrics; no learned estimator and no decision gate."""
import hashlib
import numpy as np

def group(values,seeds):
    x=np.asarray(values,dtype=float);assert len(x)==len(seeds) and np.isfinite(x).all() and len(x)>1
    low=float(x.min())
    return dict(n=len(x),seeds=list(seeds),values=x.tolist(),mean=float(x.mean()),sample_SD=float(x.std(ddof=1)),
        worst=low,worst_seeds=[s for s,v in zip(seeds,x) if v==low],
        lower_quartile=float(np.quantile(x,.25,interpolation='linear')))

def paired(x,t,seeds):
    d=np.asarray(x,dtype=float)-np.asarray(t,dtype=float)
    sd=float(d.std(ddof=1))
    return dict(seeds=list(seeds),differences=d.tolist(),mean=float(d.mean()),sample_SD=sd,
        SE=sd/np.sqrt(len(d)),positive=int((d>0).sum()),zero=int((d==0).sum()),n=len(d))

def sd_ratio(x,t,namespace,replicates=20000):
    x=np.asarray(x,dtype=float);t=np.asarray(t,dtype=float);assert len(x)==len(t)
    tx=float(t.std(ddof=1)); xx=float(x.std(ddof=1))
    result=dict(point=None if tx<=1e-12 else xx/tx,bootstrap_percentile_interval=None,
        interval_percent=95,replicates=replicates,paired_seed_resampling=True,
        method='percentile; condition on at least two distinct seed indices; undefined denominator is reported',
        seed=int.from_bytes(hashlib.sha256(('R14-bootstrap|'+namespace).encode()).digest()[:4],'little'))
    if tx<=1e-12:
        result['undefined_reason']='target SD <= 1e-12';return result
    rng=np.random.RandomState(result['seed']);ratios=[];rejected=0;undefined=0
    while len(ratios)<replicates:
        ids=rng.randint(len(x),size=(min(1024,replicates-len(ratios)),len(x)))
        valid=np.any(ids!=ids[:,[0]],axis=1);rejected+=int((~valid).sum());ids=ids[valid]
        dx=t[ids].std(axis=1,ddof=1);nx=x[ids].std(axis=1,ddof=1)
        if np.any(dx<=1e-12):
            undefined+=int((dx<=1e-12).sum())
            result.update(undefined_reason='non-degenerate bootstrap sample has target SD <= 1e-12',
                undefined_draws=undefined,degenerate_draws_rejected=rejected);return result
        ratios.extend((nx/dx).tolist())
    result.update(bootstrap_percentile_interval=np.quantile(ratios,[.025,.975],interpolation='linear').tolist(),
        degenerate_draws_rejected=rejected,undefined_draws=undefined)
    return result

def best_weight(groups):
    candidates={str(w):groups[str(w)]['mean'] for w in (0.,.25,1.)}
    maximum=max(candidates.values());tied=sorted(float(w) for w,v in candidates.items() if abs(v-maximum)<=1e-8)
    return dict(selected_w=tied[0],selected_lambda=tied[0]/(1+tied[0]),means=candidates,
        ties_within_1e8=tied,tie_rule='within 1e-8 choose smaller w',descriptive_grid_winner_only=True)
