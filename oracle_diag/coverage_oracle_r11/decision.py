"""Pre-fixed two-stage temperature/horizon choice, pure CPU arithmetic."""
import math
import statistics

SEEDS=(101,102,103)
TEMP_PREFERENCE={3.:0,1.:1,10.:2}

def choose_temperature(candidates):
    assert sorted(x['temp'] for x in candidates)==[1.,3.,10.]
    for x in candidates:
        assert len(x['runs'])==3
        assert sorted(v['job']['seed'] for v in x['runs'])==list(SEEDS)
        assert all(v['job']['temp']==x['temp'] and v['job']['steps']==500000 for v in x['runs'])
    maximum=max(x['mean'] for x in candidates)
    return min([x for x in candidates if abs(x['mean']-maximum)<=1e-8],
        key=lambda x:TEMP_PREFERENCE[x['temp']])

def choose_horizon(short_runs,long_runs,temp):
    assert len(short_runs)==len(long_runs)==3
    short=sorted(short_runs,key=lambda v:v['job']['seed'])
    long=sorted(long_runs,key=lambda v:v['job']['seed'])
    assert [x['job']['seed'] for x in short]==[x['job']['seed'] for x in long]==list(SEEDS)
    for a,b in zip(short,long):
        assert a['job']['temp']==b['job']['temp']==temp
        assert a['job']['steps']==500000 and b['job']['steps']==1000000
        assert a['last5_steps']==[460000,470000,480000,490000,500000]
        assert b['last5_steps']==[960000,970000,980000,990000,1000000]
    differences=[b['Last5']-a['Last5'] for a,b in zip(short,long)]
    mean=statistics.mean(differences)
    sd=statistics.stdev(differences)
    se=sd/math.sqrt(3)
    accept=mean>2*se
    return dict(temp=temp,steps=1000000 if accept else 500000,
        paired_differences=differences,paired_mean=mean,sample_SD=sd,SE=se,threshold=2*se,
        accepted_1M=accept,rule='strict paired_mean > 2*SE; otherwise500k',
        seeds=list(SEEDS),short_last5_steps=short[0]['last5_steps'],long_last5_steps=long[0]['last5_steps'],
        window_limit='different late-training stages, not matched absolute-step or same-stage comparison',
        intentional_short_horizon_preference=True,equivalence_claim=False,strict_95pct_claim=False,
        runs=long if accept else short,
        mean=statistics.mean(x['Last5'] for x in (long if accept else short)),
        short_runs=short,long_runs=long)
