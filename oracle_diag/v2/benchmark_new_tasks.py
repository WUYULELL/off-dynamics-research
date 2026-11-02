"""Extend B to all accepted C tasks; reuse Walker ensemble, fit target-only Hopper ensemble."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from support import *
import numpy as np,torch,time,copy
from stage_b import network,predict,healthy,corr
from common import load_target_buffer,build_bellman_agent,make_target_env,target_subsample_indices,load_policy_config
from offline_offline.iql_bellman_fb import IQL_BELLMAN_FB

def main():
    dest=OUT/'stage_b_new_tasks';fresh(dest);torch.set_num_threads(2);results={}
    accepted=read(OUT/'stage_c/completed.json')['datasets']
    for base,env in [('walker2d',ENVS[0]),('hopper',ENVS[1])]:
        tar,full=load_target_buffer(env,0);agent=build_bellman_agent(env,0,tar);dim=tar.state.shape[1]
        mean=agent.target_mean.numpy();std=agent.target_std.numpy();models=[]
        x=torch.tensor((np.concatenate([tar.state,tar.action],1)-mean)/std,dtype=torch.float32)
        if base=='walker2d':
            normalization=np.load(OUT/'stage_b/split_arrays.npz');ym=normalization['ymean'];ys=normalization['ystd'];val=~normalization['train']
            for seed in range(5):
                m=network(x.shape[1],dim+1);m.load_state_dict(torch.load(str(OUT/'stage_b'/('model_%d.pt'%seed)),map_location='cpu'));m.eval();models.append(m)
        else:
            e=make_target_env(env);raw=e.get_dataset();rawids=np.flatnonzero(~raw['timeouts'][:-1].astype(bool))
            rawids=rawids[target_subsample_indices(full,.1,0,dim,tar.action.shape[1])]
            assert np.array_equal(raw['observations'][rawids].astype(np.float32),tar.state)
            episode=np.r_[0,np.cumsum((raw['terminals']|raw['timeouts'])[:-1])][rawids]
            groups=np.unique(episode);vg=np.random.default_rng(0).permutation(groups)[:int(np.ceil(.1*len(groups)))];val=np.isin(episode,vg);train=~val
            yraw=np.concatenate([tar.next_state-tar.state,tar.reward],1);ym=yraw[train].mean(0);ys=np.maximum(yraw[train].std(0),1e-6)
            y=torch.tensor((yraw-ym)/ys,dtype=torch.float32);ti=torch.tensor(np.flatnonzero(train));vi=torch.tensor(np.flatnonzero(val))
            np.savez_compressed(dest/'hopper_split.npz',train=train,raw_indices=rawids,episode=episode,ymean=ym,ystd=ys,xmean=mean,xstd=std)
            for seed in range(5):
                torch.manual_seed(seed);m=network(x.shape[1],dim+1);opt=torch.optim.Adam(m.parameters(),lr=.001);gen=torch.Generator().manual_seed(seed)
                best=float('inf');stale=0;trace=[]
                for epoch in range(1,201):
                    m.train();order=ti[torch.randperm(len(ti),generator=gen)]
                    for batch in order.split(512):
                        loss=(m(x[batch])-y[batch]).square().mean();opt.zero_grad();loss.backward();opt.step()
                    m.eval();vl=float(np.mean((predict(m,x[vi])-y[vi].numpy())**2));trace.append(dict(epoch=epoch,val_loss=vl))
                    if vl<best:best=vl;state=copy.deepcopy(m.state_dict());best_epoch=epoch;stale=0
                    else:stale+=1
                    write(dest/'progress.json',dict(task='hopper target model',member=seed,epoch=epoch,val_loss=vl),replace=True)
                    if stale>=10:break
                m.load_state_dict(state);m.eval();models.append(m);torch.save(state,str(dest/('hopper_model_%d.pt'%seed)))
                write(dest/('hopper_member_%d.json'%seed),dict(trace=trace,best_epoch=best_epoch));print('HOPPER MODEL',seed,'done',flush=True)
            e.close()
        vp=np.array([predict(m,x[val])*ys+ym for m in models]);sstd=std[0,:dim]
        vd=np.linalg.norm(vp[:,:,:dim].std(0)/sstd,axis=1);threshold=float(np.quantile(vd,.9))
        target_validation=stats(np.linalg.norm((tar.state[val]+vp[:,:,:dim].mean(0)-tar.next_state[val])/sstd,axis=1))
        fixed=[]
        for kind in ['TARGET','MIX_FB']:
            ck=read(PREV/'diagnostics'/(env+'_'+kind+'_bellman.json'));v=IQL_BELLMAN_FB(load_policy_config(env),torch.device('cpu'))
            v.v_func.load_state_dict(torch.load(ck['checkpoint'],map_location='cpu'));v.v_func.eval();fixed.append((kind,v))
        for name,info in accepted.items():
            if not name.startswith(base) or not info['accepted']:continue
            src=dict(np.load(info['oracle_path']));sx=torch.tensor((np.concatenate([src['source_state'],src['source_action']],1)-mean)/std,dtype=torch.float32)
            sp=np.array([predict(m,sx)*ys+ym for m in models]);dis=np.linalg.norm(sp[:,:,:dim].std(0)/sstd,axis=1);gate=dis<=threshold
            mn=src['source_state']+sp[:,:,:dim].mean(0);mr=sp[:,:,dim].mean(0)
            if base=='walker2d':mnd=healthy(mn);health_note='Exact Walker2d-v2 observable height/angle rule'
            else:
                mnd=(np.isfinite(mn).all(1)&(np.abs(mn[:,1:])<100).all(1)&(mn[:,0]>.7)&(np.abs(mn[:,1])<.2)).astype(np.float32)
                health_note='Hopper-v2 healthy rule projected onto predicted observation. Clipped qvel hides whether latent velocity exceeds 100; this part cannot be exact without a latent-state model.'
            js=[]
            with torch.no_grad():
                for part in sx.split(128):_,j=agent.compute_knn_indices(part);js.append(j.numpy())
            j=np.concatenate(js);ns=agent.target_next_state_bank.numpy();s=agent.target_state_bank.numpy()
            nr=agent.target_reward_bank.numpy().ravel()[j].mean(1);nd=agent.target_not_done_bank.numpy().ravel()[j].mean(1)
            estimates={'C':(src['source_next'],src['source_reward'],src['source_not_done']),
                'A':(ns[j].mean(1),nr,nd),'B':(src['source_state']+(ns[j]-s[j]).mean(1),nr,nd),'M':(mn,mr,mnd),
                'M-gate':(np.where(gate[:,None],mn,src['source_next']),np.where(gate,mr,src['source_reward']),np.where(gate,mnd,src['source_not_done']))}
            errors={};metrics={};ce=np.linalg.norm((src['source_next']-src['oracle_next'])/sstd,axis=1)
            for key,(sn,r,nond) in estimates.items():
                e=np.linalg.norm((sn-src['oracle_next'])/sstd,axis=1);metrics[key]=dict(state=stats(e),C_state=stats(ce),better_than_C_fraction=float((e<ce).mean()),bellman={});errors[key+'_state']=e
            for kind,v in fixed:
                def backup(sn,r,nd):
                    with torch.no_grad():return (torch.tensor(r,dtype=torch.float32)+.99*torch.tensor(nd,dtype=torch.float32)*v.v_func(torch.tensor(sn,dtype=torch.float32)).ravel()).numpy()
                truth=backup(src['oracle_next'],src['oracle_reward'],src['oracle_not_done']);baseline=np.abs(backup(*estimates['C'])-truth);cs=stats(baseline)
                for key,arr in estimates.items():
                    err=np.abs(backup(*arr)-truth);st=stats(err);metrics[key]['bellman'][kind]=dict(error=st,C_error=cs,
                        mean_relative_improvement=1-st['mean']/cs['mean'] if cs['mean'] else None,
                        median_relative_improvement=1-st['median']/cs['median'] if cs['median'] else None,better_than_C_fraction=float((err<baseline).mean()));errors[key+'_'+kind]=err
            result=dict(metrics=metrics,gate_threshold=threshold,gate_fraction=float(gate.mean()),termination_note=health_note,target_validation_state_error=target_validation,
                disagreement_oracle_error_spearman=corr(dis,errors['M_state']),used_for_Walker_kinematic_candidate_selection=False,
                fixed_V_scope='Same-environment learned Vs from prior kinematic experiments; not trained on this new task')
            write(dest/(name+'.json'),result);np.savez_compressed(dest/(name+'_errors.npz'),**errors,gate=gate,disagreement=dis);results[name]=result
            print('NEW TASK BENCHMARK',name,'complete',flush=True)
    write(dest/'completed.json',dict(completed=True,tasks=results,cuda_initialized=torch.cuda.is_initialized()))
if __name__=='__main__':main()
