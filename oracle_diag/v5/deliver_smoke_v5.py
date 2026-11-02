"""CPU-only delivery audit; preserve original smoke audit and all training outputs."""
from audit_v5 import *
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def records(folder):
    return [json.loads(s) for s in (Path(folder)/'throughput.jsonl').read_text().splitlines()]


def model_difference(left, right, filename):
    p=list(Path(left).rglob(filename)); q=list(Path(right).rglob(filename))
    assert len(p)==len(q)==1
    a=torch.load(winpath(p[0]),map_location='cpu'); b=torch.load(winpath(q[0]),map_location='cpu')
    assert a.keys()==b.keys()
    return max(float((a[k]-b[k]).abs().max()) for k in a)


def main():
    dest=OUT/'smoke_delivery1'; fresh(dest)
    plan=read(OUT/'training_plan.json'); protocol=read(OUT/'protocol.json')
    jobs=[j for j in plan['jobs'] if j['kind']=='smoke']; formal=[j for j in plan['jobs'] if j['kind']=='formal']
    assert len(jobs)==10 and len(formal)==30
    assert not (OUT/'gpu_formal_authorization.json').exists()
    assert all(not Path(j['out']).exists() for j in formal)
    assert plan['protocol_sha256']==sha(OUT/'protocol.json')
    check_approval('smoke')
    original=read(OUT/'smoke_audited.json'); assert original['passed']
    rows={}; independent={}
    for j in jobs:
        row=audit_job(j); assert row==original['rows'][j['id']] or all(row[k]==original['rows'][j['id']][k] for k in row if k not in ('raw','scores'))
        folder=Path(j['out']); acc=EventAccumulator(winpath(row['tb_file']),size_guidance={'scalars':0}).Reload()
        scalars=acc.Scalars('test/target_return'); assert len(scalars)==1 and scalars[0].step==5000
        raw=float(scalars[0].value); score=normalized(raw)
        assert abs(score-row['final_score'])<2e-5
        progress=read(folder/'progress.json')
        final_training=acc.Scalars('train/value')[-1]
        assert abs(final_training.wall_time-(folder/'progress.json').stat().st_mtime)<.2
        tail=read(folder/'completed.json')['ended']-(folder/'progress.json').stat().st_mtime
        assert tail>0
        row.update(phase=j['phase'],source_draws=progress['source_draws'],target_draws=progress['target_draws'],
                   rate=progress['steps_per_second'],post_training_eval_and_save_seconds=tail)
        rows[j['id']]=row
        independent[j['id']]=dict(reader='EventAccumulator (separate from LegacyEventFileLoader)',raw=raw,
            normalized_from_raw=score,difference_from_primary=score-row['final_score'],
            progress_mtime=(folder/'progress.json').stat().st_mtime,final_training_tb_wall_time=final_training.wall_time,
            eval_tb_wall_time=scalars[0].wall_time)
        print('Independent audit passed:',j['id'],flush=True)
    a_new=next(j['out'] for j in jobs if j['method']=='ORACLE-W1'); a_old=V4_OUT/'s1/w0'
    reuseA=reuse_a(a_new)
    x=read(Path(a_new)/'early_losses.json')['events']; y=read(a_old/'early_losses.json')['events']
    reuseA['max_early_loss_abs_difference']=max(abs(a['value']-b['value']) for a,b in zip(x,y))
    reuseA['max_parameter_abs_difference']={f:model_difference(a_new,a_old,f) for f in ('model_actor','model_critic','V_5000.pt')}
    reuseA['normalized_eval_abs_difference']=abs(events(a_new)[1]['test/target_normalized_score'][5000]-events(a_old)[1]['test/target_normalized_score'][5000])
    b_new=next(j['out'] for j in jobs if j['method']=='OTDF'); reuseB=reuse_b(b_new)
    init=read(Path(b_new)/'initial_evaluation.json'); reuseB['initial_return_abs_differences']=[abs(init[k]-v) for k,v in zip(('source','target'),reuseB['initial_returns'])]
    reuseB['first_vae_abs_difference']=abs(reuseB['first_vae']-events(reuseB['old'])[1]['train/VAE_loss'][1])
    concurrent={j['stack']:j for j in jobs if j['phase']=='concurrent'}
    # Reconstruct per-block wall times from the final progress timestamp, then
    # retain only block endpoints inside BOTH recorded policy-training spans.
    starts={}; ends={}; origins={}
    for s,j in concurrent.items():
        folder=Path(j['out']); p=read(folder/'progress.json')
        ends[s]=(folder/'progress.json').stat().st_mtime
        starts[s]=(folder/('first_sample.json' if s=='A' else 'initial_evaluation.json')).stat().st_mtime
        origins[s]=ends[s]-p['elapsed']
    start=max(starts.values()); end=min(ends.values()); assert end>start
    overlap={}; concurrent_models={}
    for s,j in concurrent.items():
        solo=next(k for k in jobs if k['phase']=='standalone' and k['stack']==s and k['method']==j['method'])
        pts=[p for p in records(j['out']) if start<=origins[s]+p['elapsed']<=end]
        assert len(pts)>=2
        lo,hi=pts[0],pts[-1]; solo_pts={p['step']:p for p in records(solo['out'])}
        count=hi['step']-lo['step']; wall=hi['elapsed']-lo['elapsed']; solo_wall=solo_pts[hi['step']]['elapsed']-solo_pts[lo['step']]['elapsed']
        overlap[s]=dict(method=j['method'],steps=[lo['step'],hi['step']],seconds=wall,standalone_rate=count/solo_wall,
            concurrent_rate=count/wall,ratio=solo_wall/wall,block_epochs=[origins[s]+lo['elapsed'],origins[s]+hi['elapsed']])
        assert overlap[s]['ratio']>=.5
        concurrent_models[s]={f:model_difference(j['out'],solo['out'],f) for f in ('model_actor','model_critic')}
        assert all(v==0 for v in concurrent_models[s].values())
    eta={}; method_eta={}; wall_eta={}
    for s,methods in [('A',A_METHODS),('B',B_METHODS)]:
        ratio=min(1.,overlap[s]['ratio']); steps=protocol[s]['steps']; method_eta[s]={}; wall_eta[s]=0.
        for m in methods:
            j=next(j for j in jobs if j['method']==m and j['phase']=='standalone'); folder=Path(j['out']); p=read(folder/'progress.json')
            vae=read(folder/'vae_completed.json')['elapsed'] if (folder/'vae_completed.json').exists() else 0.
            setup=p['elapsed']-p['training_seconds']-vae
            tail=rows[j['id']]['post_training_eval_and_save_seconds']
            # Once-only setup/VAE; 50 or 100 evaluation+save blocks. Tail also
            # contains final writes, so this is an approximate timing proxy.
            train_seconds=steps/p['steps_per_second']/ratio
            eval_seconds=(steps//10000)*tail/ratio
            one=setup+vae+train_seconds+eval_seconds
            method_eta[s][m]=dict(setup_seconds=setup,vae_seconds=vae,eval_and_save_proxy_seconds=tail,
                concurrent_ratio_capped=ratio,training_hours_per_seed=train_seconds/3600,
                eval_save_hours_per_seed=eval_seconds/3600,hours_per_seed=one/3600,hours_five_seeds=5*one/3600)
            wall_eta[s]+=5*(steps/(5000/(p['elapsed']-vae))+vae)/ratio/3600
        eta[s]=sum(v['hours_five_seeds'] for v in method_eta[s].values())
    a5=read(OUT/'a5_2/completed.json'); assert a5['passed']
    a5_summary={m:{k:dict(mean=statistics.mean([r[k] for r in a5['rows'] if r['method']==m]),
                       sd=statistics.stdev([r[k] for r in a5['rows'] if r['method']==m]))
        for k in ('target_action_mse','source_neighbor_action_mse')} for m in ('TARGET-64','ORACLE-W1','ORACLE-ONLY')}
    count=protected_check(); assert count==3037
    assert not torch.cuda.is_initialized()
    code_hash=read(OUT/'cpu_ready.json')['code_sha256']
    assert all(sha(p)==h for p,h in code_hash.items())
    input_files=[OUT/f for f in ('protocol.json','training_plan.json','gpu_smoke_authorization.json','cpu_ready.json',
         'smoke_audited.json','protected_manifest.json','existing_baselines.json','smoke_pipeline_status.json')]
    for j in jobs:
        input_files += [Path(j['out'])/f for f in ('completed.json','progress.json','throughput.jsonl','effective_config.json',
            'initialization.json','data_verified.json','evaluation_verified.json') if (Path(j['out'])/f).exists()]
    result=dict(passed=True,created=time.time(),scope='GPU smoke delivery only; no formal authorization',jobs=rows,
       independent_tensorboard=independent,reuse_A=reuseA,reuse_B=reuseB,overlap_interval=dict(start=start,end=end,seconds=end-start,
       starts=starts,ends=ends,method='file mtime anchored cumulative wall time; TB validates final anchors'),overlap=overlap,
       concurrent_model_max_differences=concurrent_models,formal_eta_hours=eta,method_eta=method_eta,
       alternate_wall_projection_hours=wall_eta,original_automated_eta_hours=original['conservative_eta_hours'],
       B_over_72h=eta['B']>72,formal_jobs=30,formal_started=False,protected_files=count,frozen_code_files=len(code_hash),
       a5_summary=a5_summary,input_sha256={str(p):sha(p) for p in input_files},delivery_script_sha256=sha(__file__),
       timing_limits=['single short overlapping trial','REG not measured concurrently','future policy evaluation episode lengths can change',
          'same slowdown applied to all methods and eval proxies; not a confidence interval or guaranteed upper bound'])
    write(dest/'verified.json',result)
    print(json.dumps(dict(eta=eta,alternate_wall_projection=wall_eta,overlap=overlap,reuseA=reuseA,reuseB=reuseB),ensure_ascii=True),flush=True)


if __name__=='__main__':main()
