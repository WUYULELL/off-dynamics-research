"""Trajectory-held-out target-only dynamics ensemble and fixed-V correction benchmark."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from support import *
import time,copy,inspect
import numpy as np,torch
from scipy.stats import spearmanr
from common import load_target_buffer,target_subsample_indices,make_target_env,build_bellman_agent,load_policy_config
from offline_offline.iql_bellman_fb import IQL_BELLMAN_FB

def healthy(next_state):
    # Exact installed gym Walker2d-v2 step condition; not a learned classifier.
    return ((next_state[:,0]>.8)&(next_state[:,0]<2.)&(next_state[:,1]>-1.)&(next_state[:,1]<1.)).astype(np.float32)
def network(d,o):
    return torch.nn.Sequential(torch.nn.Linear(d,200),torch.nn.ReLU(),torch.nn.Linear(200,200),torch.nn.ReLU(),torch.nn.Linear(200,200),torch.nn.ReLU(),torch.nn.Linear(200,o))
def predict(m,x):
    with torch.no_grad():return torch.cat([m(x[i:i+2048]) for i in range(0,len(x),2048)]).numpy()
def corr(a,b):
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    return float(spearmanr(a,b).correlation)

def main():
    dest=OUT/'stage_b';fresh(dest);torch.set_num_threads(2)
    env=ENVS[0];src=dict(np.load(OUT/'stage_a'/(env+'_source_oracle_fixed.npz')))
    tar,full=load_target_buffer(env,0);agent=build_bellman_agent(env,0,tar)
    gymenv=make_target_env(env);raw=gymenv.get_dataset()
    rawids=np.flatnonzero(~raw['timeouts'][:-1].astype(bool))
    rawids=rawids[target_subsample_indices(full,.1,0,tar.state.shape[1],tar.action.shape[1])]
    assert np.array_equal(raw['observations'][rawids].astype(np.float32),tar.state)
    episode=np.r_[0,np.cumsum((raw['terminals']|raw['timeouts'])[:-1])]
    groups=episode[rawids];unique=np.unique(groups);rng=np.random.default_rng(0)
    val_groups=rng.permutation(unique)[:int(np.ceil(.1*len(unique)))];val=np.isin(groups,val_groups);train=~val
    assert not set(groups[val])&set(groups[train])
    mean=agent.target_mean.numpy();std=agent.target_std.numpy()
    x=torch.tensor((np.concatenate([tar.state,tar.action],axis=1)-mean)/std,dtype=torch.float32)
    yraw=np.concatenate([tar.next_state-tar.state,tar.reward],axis=1)
    ym=yraw[train].mean(0);ys=np.maximum(yraw[train].std(0),1e-6)
    y=torch.tensor((yraw-ym)/ys,dtype=torch.float32);ti=torch.tensor(np.flatnonzero(train));vi=torch.tensor(np.flatnonzero(val))
    sx=torch.tensor((np.concatenate([src['source_state'],src['source_action']],axis=1)-mean)/std,dtype=torch.float32)
    write(dest/'split.json',dict(seed=0,train_n=int(train.sum()),validation_n=int(val.sum()),train_episodes=len(np.unique(groups[train])),
        validation_episodes=len(val_groups),group_overlap=0,validation_episode_ids=val_groups.tolist(),
        feature_standardization='Original target feature bank mean/std; output normalization uses train rows only',
        termination_rule=inspect.getsource(type(gymenv.unwrapped).step),
        hyperparameters=read(OUT/'preregistered_protocol.json')['B']))
    np.savez_compressed(dest/'split_arrays.npz',raw_indices=rawids,train=train,episode_id=groups,xmean=mean,xstd=std,ymean=ym,ystd=ys)
    models=[];traces=[];begin=time.time()
    for seed in range(5):
        torch.manual_seed(seed);m=network(x.shape[1],y.shape[1]);opt=torch.optim.Adam(m.parameters(),lr=.001)
        best=float('inf');best_state=None;stale=0;trace=[]
        generator=torch.Generator().manual_seed(seed)
        for epoch in range(1,201):
            order=ti[torch.randperm(len(ti),generator=generator)];m.train();loss_sum=0.
            for batch in order.split(512):
                loss=(m(x[batch])-y[batch]).square().mean();opt.zero_grad();loss.backward();opt.step();loss_sum+=float(loss)*len(batch)
            m.eval();vp=predict(m,x[vi]);vl=float(np.mean((vp-y[vi].numpy())**2));trace.append(dict(epoch=epoch,train=loss_sum/len(ti),validation=vl))
            if vl<best:
                best=vl;best_state=copy.deepcopy(m.state_dict());best_epoch=epoch;stale=0
            else:stale+=1
            write(dest/'progress.json',dict(member=seed,epoch=epoch,val_loss=vl,best_val_loss=best,elapsed=time.time()-begin),replace=True)
            if epoch%10==0:print('MODEL',seed,'epoch',epoch,'validation',vl,'elapsed',time.time()-begin,flush=True)
            if stale>=10:break
        m.load_state_dict(best_state);m.eval();torch.save(best_state,str(dest/('model_%d.pt'%seed)));models.append(m)
        traces.append(dict(seed=seed,best_epoch=best_epoch,best_validation_mse=best,epochs=trace))
        write(dest/('member_%d.json'%seed),traces[-1])
    sp=np.array([predict(m,sx)*ys+ym for m in models]);vp=np.array([predict(m,x[vi])*ys+ym for m in models])
    state_dim=src['source_state'].shape[1];sstd=src['target_state_std']
    disagreement=np.linalg.norm(sp[:,:,:state_dim].std(0)/sstd,axis=1)
    vd=np.linalg.norm(vp[:,:,:state_dim].std(0)/sstd,axis=1);threshold=float(np.quantile(vd,.9));gate=disagreement<=threshold
    mn=src['source_state']+sp[:,:,:state_dim].mean(0);mr=sp[:,:,state_dim].mean(0);mnd=healthy(mn)
    valn=tar.state[val]+vp[:,:,:state_dim].mean(0);valr=vp[:,:,state_dim].mean(0)
    # Reuse old A/B arrays by original IDs; only the new last row needs a query.
    geo=np.load(PREV/'diagnostics'/(env+'_geometry.npz'));assert np.array_equal(agent.target_state_bank.numpy(),geo['bank_state'])
    n=len(sx);an=np.empty_like(src['source_next']);bn=np.empty_like(an);nr=np.empty(n);nnd=np.empty(n)
    ids=geo['query_indices'];an[ids]=geo['pred_a'];bn[ids]=geo['pred_b'];j=geo['knn_indices']
    nr[ids]=geo['bank_reward'][j].mean(1);nnd[ids]=geo['bank_not_done'][j].mean(1)
    with torch.no_grad():_,newj=agent.compute_knn_indices(sx[-1:]);newj=newj.numpy()
    an[-1]=geo['bank_next'][newj].mean((0,1));bn[-1]=src['source_state'][-1]+(geo['bank_next'][newj]-geo['bank_state'][newj]).mean((0,1))
    nr[-1]=geo['bank_reward'][newj].mean();nnd[-1]=geo['bank_not_done'][newj].mean()
    estimates={'C':(src['source_next'],src['source_reward'],src['source_not_done']),'A':(an,nr,nnd),'B':(bn,nr,nnd),'M':(mn,mr,mnd),
        'M-gate':(np.where(gate[:,None],mn,src['source_next']),np.where(gate,mr,src['source_reward']),np.where(gate,mnd,src['source_not_done']))}
    metrics={};errors={};truth=src['oracle_next'];ce=np.linalg.norm((src['source_next']-truth)/sstd,axis=1)
    for key,(ns,r,nd) in estimates.items():
        error=np.linalg.norm((ns-truth)/sstd,axis=1);metrics[key]=dict(state=stats(error),better_than_C_fraction=float((error<ce).mean()),bellman={});errors[key+'_state']=error
    for kind in ['TARGET','MIX_FB']:
        ck=read(PREV/'diagnostics'/(env+'_'+kind+'_bellman.json'))
        m=IQL_BELLMAN_FB(load_policy_config(env),torch.device('cpu'));m.v_func.load_state_dict(torch.load(ck['checkpoint'],map_location='cpu'));m.v_func.eval()
        def backup(ns,r,nd):
            with torch.no_grad():return (torch.as_tensor(r,dtype=torch.float32)+.99*torch.as_tensor(nd,dtype=torch.float32)*m.v_func(torch.as_tensor(ns,dtype=torch.float32)).ravel()).numpy()
        yo=backup(src['oracle_next'],src['oracle_reward'],src['oracle_not_done']);baseline=np.abs(backup(*estimates['C'])-yo)
        for key,arr in estimates.items():
            e=np.abs(backup(*arr)-yo);d=stats(e);c=stats(baseline)
            metrics[key]['bellman'][kind]=dict(error=d,C_error=c,mean_relative_improvement=1-d['mean']/c['mean'],median_relative_improvement=1-d['median']/c['median'],better_than_C_fraction=float((e<baseline).mean()))
            errors[key+'_'+kind]=e
    passed=[k for k in ['M-gate','M','B','A'] if all(v['mean_relative_improvement']>=.2 and v['median_relative_improvement']>=.2 for v in metrics[k]['bellman'].values())]
    selected=passed[0] if passed else None
    if selected:
        ns,r,nd=estimates[selected];arr=dict(src);arr.update(oracle_next=ns,oracle_reward=r,oracle_not_done=nd)
        np.savez_compressed(dest/'selected_correction_fixed.npz',**arr)
    np.savez_compressed(dest/'diagnostic_arrays.npz',**errors,gate=gate,disagreement=disagreement,validation_disagreement=vd)
    for key,arr in estimates.items():np.savez_compressed(dest/('estimate_'+key+'.npz'),next_state=arr[0],reward=arr[1],not_done=arr[2])
    result=dict(completed=True,metrics=metrics,gate_threshold=threshold,source_gate_fraction=float(gate.mean()),
        disagreement_oracle_state_error_spearman=corr(disagreement,errors['M_state']),
        target_validation=dict(state_error=stats(np.linalg.norm((valn-tar.next_state[val])/sstd,axis=1)),reward_abs=stats(np.abs(valr-tar.reward[val].ravel()))),
        passed_candidates=passed,selected=selected,selection_rule='Precommitted M-gate, M, B, A order among candidates satisfying both V gates',
        selected_file=str(dest/'selected_correction_fixed.npz') if selected else None,
        elapsed=time.time()-begin,cuda_initialized=torch.cuda.is_initialized())
    write(dest/'completed.json',result);gymenv.close();print('STAGE B COMPLETE',selected,flush=True)
if __name__=='__main__':main()
