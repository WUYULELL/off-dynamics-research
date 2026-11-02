import math,statistics
def group(values,seeds):
    a=list(values);assert len(a)==len(seeds) and len(a)>=2 and all(math.isfinite(v) for v in a)
    b=sorted(a);f=(len(a)-1)*.25;lo=int(f);hi=math.ceil(f)
    return dict(n=len(a),seeds=seeds,values=a,mean=statistics.mean(a),sample_SD=statistics.stdev(a),worst=min(a),worst_seeds=[s for s,v in zip(seeds,a) if v==min(a)],Q1=b[lo]+(b[hi]-b[lo])*(f-lo))
def welch(a,b):
    return dict(mean_difference=a['mean']-b['mean'],SE=math.sqrt(a['sample_SD']**2/a['n']+b['sample_SD']**2/b['n']),comparison='unpaired Welch standard error')
def gate(H,SE):
    if H>=10 and H>2*SE:return '估分员是瓶颈，存在提升空间'
    if H<5:return '该结构下提升空间不足'
    if 5<=H<10:return '不确定'
    return 'V9未覆盖：H>=10但未超过2SE，停止解释并请用户决定'
