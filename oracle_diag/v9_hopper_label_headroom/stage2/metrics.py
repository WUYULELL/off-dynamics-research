"""V9 stage2's fixed per-weight paired rules; no bootstrap or new decision gates."""
import numpy as np

def group(values,seeds):
    a=np.asarray(values,dtype=np.float64);assert len(a)==len(seeds)==10 and np.isfinite(a).all()
    return dict(n=10,seeds=list(seeds),values=a.tolist(),mean=float(a.mean()),sample_SD=float(a.std(ddof=1)),
        worst=float(a.min()),worst_seeds=[s for s,v in zip(seeds,a) if v==a.min()],Q1=float(np.quantile(a,.25,interpolation='linear')))

def paired(oracle,reference,seeds):
    d=np.asarray(oracle,dtype=np.float64)-np.asarray(reference,dtype=np.float64)
    assert len(d)==len(seeds)==10
    sd=float(d.std(ddof=1))
    return dict(n=10,seeds=list(seeds),differences=d.tolist(),mean=float(d.mean()),sample_SD=sd,
        SE=float(sd/np.sqrt(10)),positive=int((d>0).sum()),zero=int((d==0).sum()))

def decision(L,E):
    conditions=dict(L_mean_strict_gt_2SE=L['mean']>2*L['SE'],L_positive_ge7=L['positive']>=7,
        E_mean_strict_gt_minus2SE=E['mean']>-2*E['SE'],E_mean_strict_lt_minus2SE=E['mean']<-2*E['SE'],
        E_mean_strict_gt_2SE=E['mean']>2*E['SE'],E_positive_ge7=E['positive']>=7)
    A=conditions['L_mean_strict_gt_2SE'] and conditions['L_positive_ge7'] and conditions['E_mean_strict_gt_minus2SE']
    B=not conditions['L_mean_strict_gt_2SE'] and conditions['E_mean_strict_lt_minus2SE']
    assert not (A and B)
    state='A' if A else 'B' if B else 'uncertain'
    return dict(rule_version='V9-stage2-verbatim-strict-boundaries',category=state,
        label={'A':'伤害主要来自标签偏差','B':'伤害不来自标签','uncertain':'不确定'}[state],conditions=conditions,
        corrected_source_gain=bool(A and conditions['E_mean_strict_gt_2SE'] and conditions['E_positive_ge7']),
        thresholds_are_user_prespecified_descriptive_rules=True,formal_95percent_or_equivalence_claim=False)
