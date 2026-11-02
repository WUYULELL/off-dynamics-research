"""Read original scalar event records and CPU model tensors; independent of training logs."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from common_v5 import *
import argparse, ast, math, re, statistics
import numpy as np
import torch
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader

def events(folder):
    files=list(Path(folder).rglob('events.out.tfevents.*'));assert len(files)==1,files
    values={}
    for e in LegacyEventFileLoader(winpath(files[0])).Load():
        for v in e.summary.value:
            if v.WhichOneof('value')!='simple_value':continue
            assert math.isfinite(v.simple_value),v.tag
            series=values.setdefault(v.tag.replace(' ','_'),{});assert e.step not in series,(v.tag,e.step)
            series[e.step]=float(v.simple_value)
    return files[0],values

def finite(value):
    if torch.is_tensor(value):assert torch.isfinite(value).all()
    elif isinstance(value,dict):
        for v in value.values():finite(v)
    elif isinstance(value,(list,tuple)):
        for v in value:finite(v)
    elif isinstance(value,float):assert math.isfinite(value)

def paired(values):
    assert len(values)==5 and all(math.isfinite(x) for x in values)
    avg=statistics.mean(values);sd=statistics.stdev(values);se=sd/math.sqrt(5);positive=sum(x>0 for x in values)
    return dict(values=values,mean=avg,sample_sd=sd,se=se,positive=positive,passed=avg>2*se and positive>=4)

def normalized(raw):
    c=read(OUT/'protocol.json')['normalization'];return (raw-c['min'])/(c['max']-c['min'])*100

def old_b_config():
    run=read(ROOT/'reproductions/otdf_setup/study_five_seeds.json')['seeds'][0]['run_dir']
    return dict(ast.literal_eval(s.strip()) for s in Path(run+'log.txt').read_text().splitlines() if s.strip().startswith('('))

def audit_job(j):
    folder=Path(j['out']);done=read(folder/'completed.json');cfg=read(folder/'effective_config.json');c=cfg['config']
    assert all(done[k]==j[k] for k in ('method','seed','steps'))
    assert cfg['method']==j['method'] and cfg['seed']==j['seed'] and cfg['purpose']==j['kind']
    assert cfg['protocol_sha256']==sha(OUT/'protocol.json')
    progress=read(folder/'progress.json');assert progress['step']==j['steps']
    f,sc=events(folder);raw=sc['test/target_return'];expected=[5000] if j['kind']=='smoke' else list(range(10000,j['steps']+1,10000))
    assert sorted(raw)==expected
    ev=read(folder/'evaluation_verified.json')
    assert {k:ev['normalization'][k] for k in ('min','max')}==read(OUT/'protocol.json')['normalization']
    evaluated={r['step']:r for r in ev['events']};assert sorted(evaluated)==([0]+expected if j['stack']=='B' else expected)
    scores={step:normalized(raw[step]) for step in expected}
    xml=sha(ROOT/'reproductions/OTDF/envs/assets/walker_morph.xml')
    for step in expected:
        row=evaluated[step]
        assert row['confirmed_target_env_object'] and row['time_limit']==1000 and row['xml_sha256']==xml
        assert row['env_class']=='gym.envs.mujoco.walker2d_v3'
        assert abs(normalized(row['raw_return'])-scores[step])<2e-5
    data=read(folder/'data_verified.json');assert data['source_n']==999322
    if j['stack']=='A':
        old=V4_OUT/(('s1/w0' if j['kind']=='smoke' else 'b1/w'+str(j['seed'])) )
        base=read(old/'effective_config.json')['config'];expected_config=dict(base,seed=j['seed'])
        assert c==expected_config,(c,expected_config)
        assert progress['source_draws']==progress['target_draws']==64*j['steps'] and progress['q_calls']==2
        assert data['training_buffer_n']==5000 and data['all_source_rows_aligned'] and data['routed_data']['all_rows_equal']
        assert read(folder/'initialization.json')['sha256']==read(V4_OUT/('b1/w'+str(j['seed']))/'initialization.json')['sha256']
        first=read(folder/'first_sample.json');rng=np.random.RandomState(j['seed'])
        assert first['source_ids']==rng.randint(0,999322,64).tolist() and first['target_ids']==rng.randint(0,5000,64).tolist()
        weights=sc['train/source_weight_mean'];assert sorted(weights)==[1]+list(range(5000,j['steps']+1,5000)) and all(x==1. for x in weights.values())
        assert sorted(sc['test/target_normalized_score'])==expected
        for step in expected:assert abs(sc['test/target_normalized_score'][step]-scores[step])<2e-5
        # A's primary value uses the original normalized TensorBoard scalar.
        scores=sc['test/target_normalized_score']
    else:
        noreg=j['method'] in ('IQL-TARGET','OTDF-NOREG');target_only=j['method'].startswith('IQL-TARGET')
        assert c==dict(old_b_config(),noreg=noreg)
        assert data['target_n']==4999 and data['all_rows_equal'] and data['cost_rows_aligned']
        assert progress['source_draws']==(0 if target_only else j['steps']*128)
        assert progress['target_draws']==(j['steps']+(0 if noreg else 10000))*128
        assert cfg['effective_batch']==(128 if target_only else 230)
        if noreg:assert 'train/VAE_loss' not in sc and not (folder/'vae_completed.json').exists()
        else:
            assert sorted(sc['train/VAE_loss'])==list(range(1,10000,1000))
            assert read(folder/'vae_completed.json')['steps']==10000
            assert abs(sc['train/VAE_loss'][1]-read(folder/'first_vae_loss.json')['value'])<1e-5
    trace=read(folder/'early_losses.json')['events'];assert [(x['step'],x['loss']) for x in trace]==[(s,k) for s in range(1,11) for k in ('V','Q','actor')]
    assert all(math.isfinite(x['value']) for x in trace)
    files=[p for p in folder.rglob('model*') if os.path.isfile(winpath(p))];assert len(files)==7
    extras=list(folder.glob('values/*.pt'))+list(folder.glob('vae_final.pt'));hashes={}
    for p in files+extras:finite(torch.load(winpath(p),map_location='cpu'));hashes[str(p)]=sha(p)
    row=dict(method=j['method'],seed=j['seed'],stack=j['stack'],out=str(folder),raw=raw,scores=scores,final_score=scores[j['steps']],
      tb_file=str(f),tb_sha256=sha(f),artifact_sha256=hashes,elapsed=progress['elapsed'],training_seconds=progress['training_seconds'])
    if j['kind']=='formal':
        row['last5']=[scores[s] for s in read(OUT/'protocol.json')[j['stack']]['last5_steps']];row['last5_mean']=statistics.mean(row['last5'])
    return row

def reuse_a(folder):
    old=V4_OUT/'s1/w0';folder=Path(folder);protocol=read(OUT/'protocol.json')['tolerances']
    for file in ('initialization.json','first_sample.json'):
        left=read(folder/file);right=read(old/file);right.pop('loader',None);assert left==right,file
    assert read(folder/'effective_config.json')['config']==read(old/'effective_config.json')['config']
    a=read(folder/'early_losses.json')['events'];b=read(old/'early_losses.json')['events']
    for x,y in zip(a,b):assert (x['step'],x['loss'])==(y['step'],y['loss']) and np.isclose(x['value'],y['value'],**protocol['A_early_losses'])
    for filename in ('model_actor','model_critic','V_5000.pt'):
        new=list(folder.rglob(filename));prior=list(old.rglob(filename));assert len(new)==len(prior)==1
        x=torch.load(winpath(new[0]),map_location='cpu');y=torch.load(winpath(prior[0]),map_location='cpu')
        assert x.keys()==y.keys()
        for key in x:assert torch.allclose(x[key],y[key],**protocol['A_parameters']),key
    _,x=events(folder);_,y=events(old)
    assert abs(x['test/target_normalized_score'][5000]-y['test/target_normalized_score'][5000])<protocol['A_normalized_eval_abs']
    return dict(passed=True,old=str(old),new=str(folder))

def reuse_b(folder):
    folder=Path(folder);old=read(ROOT/'reproductions/otdf_setup/study_five_seeds.json')['seeds'][0]
    assert read(folder/'effective_config.json')['config']==old_b_config()
    text=Path(old['stdout_path']).read_text(encoding='utf-8',errors='replace')
    values=[float(x) for x in re.findall(r'\[0\] Evaluation over 10 episodes: ([^\s]+)',text)];assert len(values)==2
    new=read(folder/'initial_evaluation.json');t=read(OUT/'protocol.json')['tolerances']
    assert abs(new['source']-values[0])<t['B_initial_return_abs'] and abs(new['target']-values[1])<t['B_initial_return_abs']
    _,sc=events(old['run_dir']);value=read(folder/'first_vae_loss.json')['value']
    assert np.isclose(value,sc['train/VAE_loss'][1],**t['B_first_vae'])
    return dict(passed=True,old=old['run_dir'],new=str(folder),initial_returns=values,first_vae=value)

def main():
    p=argparse.ArgumentParser();p.add_argument('--job-id',required=True);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    j=next(j for j in read(OUT/'training_plan.json')['jobs'] if j['id']==a.job_id);result=audit_job(j)
    if j['method']=='ORACLE-W1':result['reuse']=reuse_a(j['out'])
    if j['method']=='OTDF':result['reuse']=reuse_b(j['out'])
    assert not torch.cuda.is_initialized();write(a.out,dict(passed=True,job=j['id'],result=result))
    print('AUDIT PASSED',j['id'])

if __name__=='__main__':main()
