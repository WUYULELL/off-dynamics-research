"""CPU audit of v3 GPU runs, including independent raw TensorBoard score checks."""
from support_v3 import *
import math,statistics,argparse

def paired(values):
    n=len(values);sd=statistics.stdev(values);se=sd/math.sqrt(n);mean=statistics.mean(values);positive=sum(x>0 for x in values)
    return dict(values=values,n=n,mean=mean,sample_sd=sd,se=se,positive=positive,
        decision='positive_gate' if mean>2*se and positive>=4 else ('limited' if mean>0 else 'nonpositive_observed_mean'))

def audit(kind):
    import numpy as np,torch
    from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader
    jobs=[j for j in read(OUT/'training_plan.json')['jobs'] if j['kind']==kind]
    assert len(jobs)==(3 if kind=='smoke' else 15)
    result=[];by_seed={}
    for j in jobs:
        folder=Path(j['out']);completed=read(folder/'completed.json');config=read(folder/'effective_config.json')
        assert all(completed[k]==j[k] for k in ['method','seed','steps'])
        assert config['total_batch']==128 and config['standard_direction'] and config['config']['source_budget']==.2
        data=read(folder/'data_verified.json');progress=read(folder/'progress.json');first=read(folder/'first_sample.json')
        assert data['source_n']==999322 and data['target_n']==5000 and data['all_source_rows_aligned'] and data['all_target_rows_equal_file']
        assert progress['step']==j['steps']
        draws=(0,128) if j['method']=='TARGET' else (64,64)
        assert (progress['source_draws'],progress['target_draws'])==tuple(x*j['steps'] for x in draws)
        files=list(folder.rglob('events.out.tfevents.*'));assert len(files)==1
        raw={};normalized={}
        for event in LegacyEventFileLoader(winpath(files[0])).Load():
            for value in event.summary.value:
                tag=value.tag.replace(' ', '_')
                if tag not in ['test/target_return','test/target_normalized_score']:continue
                assert value.WhichOneof('value')=='simple_value' and math.isfinite(value.simple_value)
                dest=raw if tag=='test/target_return' else normalized
                assert event.step not in dest;dest[event.step]=float(value.simple_value)
        steps=[5000] if kind=='smoke' else list(range(10000,500001,10000))
        assert sorted(raw)==sorted(normalized)==steps
        checks=read(folder/'evaluation_verified.json');assert [e['step'] for e in checks['events']]==steps
        const=checks['normalization']
        for e in checks['events']:
            step=e['step'];score=(raw[step]-const['min'])/(const['max']-const['min'])*100
            assert abs(score-normalized[step])<2e-5
            assert abs(e['normalized_score']-normalized[step])<2e-5 and e['confirmed_target_env_object']
        checkpoint_paths=[p for p in folder.rglob('model*') if os.path.isfile(winpath(p))]
        assert len(checkpoint_paths)==7
        def finite(x):
            if torch.is_tensor(x):assert torch.isfinite(x).all()
            elif isinstance(x,dict):
                for v in x.values():finite(v)
            elif isinstance(x,(list,tuple)):
                for v in x:finite(v)
            elif isinstance(x,float):assert math.isfinite(x)
        for path in checkpoint_paths:finite(torch.load(winpath(path),map_location='cpu'))
        row=dict(method=j['method'],seed=j['seed'],steps=j['steps'],tb_file=str(files[0]),tb_sha256=sha(files[0]),
            scores=normalized,final_score=normalized[j['steps']],elapsed=progress['elapsed'],
            last5=[normalized[s] for s in range(460000,500001,10000)] if kind=='formal' else None)
        if kind=='formal':row['last5_mean']=statistics.mean(row['last5'])
        result.append(row)
        by_seed.setdefault(j['seed'],{})[j['method']]=dict(initialization=read(folder/'initialization.json')['sha256'],first=first,data=data)
    for seed,methods in by_seed.items():
        assert len(set(m['initialization'] for m in methods.values()))==1
        assert methods['MIX-CONST']['first']==methods['MIX-ORACLE-BACKUP']['first']
        assert len(set(m['data']['source_dataset_sha256'] for m in methods.values()))==1
        assert len(set(m['data']['target_dataset_sha256'] for m in methods.values()))==1
    answer=dict(passed=True,kind=kind,runs=result,initializations_and_mixed_samplers_paired=True)
    if kind=='formal':
        values={(r['method'],r['seed']):r['last5_mean'] for r in result}
        answer['comparisons']={a+' minus '+b:paired([values[a,s]-values[b,s] for s in range(5)]) for a,b in
            [('MIX-ORACLE-BACKUP','TARGET'),('MIX-CONST','TARGET'),('MIX-ORACLE-BACKUP','MIX-CONST')]}
    else:
        answer['estimated_formal_hours_from_smoke']=sum(r['elapsed'] for r in result)*100*5/3600
        answer['timing_limit']='Crude extrapolation includes smoke initialization and excludes final evaluation; no runtime guarantee'
    return answer

def main():
    p=argparse.ArgumentParser();p.add_argument('--kind',choices=['smoke','formal'],required=True);a=p.parse_args()
    write(OUT/(a.kind+'_audited.json'),audit(a.kind))

if __name__=='__main__':main()
