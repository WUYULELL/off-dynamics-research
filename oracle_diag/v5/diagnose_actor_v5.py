"""A5: describe existing actors on all target states; no updates or rollouts."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from support_v5 import *
from data_v4 import load_source_cpu
from task_env import call_target_dataset
from offline_offline.iql import Policy
import argparse, statistics
import numpy as np
import torch
from scipy.spatial import cKDTree

def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=OUT/'a5_1');a=p.parse_args();fresh(a.out)
    torch.set_num_threads(2);started=time.time()
    src=load_source_cpu();tar=call_target_dataset('walker2d-morph')
    mean=tar['observations'].astype(np.float64).mean(0);std=tar['observations'].astype(np.float64).std(0)+1e-6
    sn=(src['observations']-mean)/std;tn=(tar['observations']-mean)/std
    print('Building exact source-state KD tree',flush=True)
    tree=cKDTree(sn);dist,ids=tree.query(tn,k=5,eps=0)
    source_actions=src['actions'][ids].astype(np.float64).mean(1)
    # Verify 12 pre-fixed query points independently by exhaustive Euclidean distance.
    for i in np.random.RandomState(20261013).choice(5000,12,replace=False):
        ds=((sn-tn[i])**2).sum(1);expected=np.sort(np.partition(ds,4)[:5])
        assert np.allclose(dist[i]**2,expected,rtol=1e-10,atol=1e-10)
    np.savez_compressed(winpath(a.out/'knn_reference.npz'),indices=ids,distances=dist,source_mean_actions=source_actions,target_mean=mean,target_std=std)
    rows=[];arrays={}
    for method,prefix in [('TARGET-64','h'),('ORACLE-W1','w'),('ORACLE-ONLY','o')]:
        for seed in range(5):
            folder=V4_OUT/('b1/'+prefix+str(seed));assert read(folder/'completed.json')['steps']==500000
            files=list(folder.rglob('model_actor'));assert len(files)==1
            model=Policy(17,6,1.,hidden_size=256).cpu()
            model.load_state_dict(torch.load(winpath(files[0]),map_location='cpu'));model.eval()
            with torch.no_grad():
                # bc_loss uses tanh(mu); compute directly to avoid sampling an unused action.
                actions=torch.cat([torch.tanh(model.network(torch.as_tensor(tar['observations'][i:i+512],dtype=torch.float32)).chunk(2,dim=1)[0]) for i in range(0,5000,512)]).numpy()
            assert np.isfinite(actions).all()
            target_mse=float(np.mean((actions.astype(np.float64)-tar['actions'])**2))
            source_mse=float(np.mean((actions.astype(np.float64)-source_actions)**2))
            rows.append(dict(method=method,seed=seed,target_action_mse=target_mse,source_neighbor_action_mse=source_mse,model=str(files[0]),sha256=sha(files[0])))
            arrays[prefix+str(seed)]=actions
            print(method,seed,target_mse,source_mse,flush=True)
    np.savez_compressed(winpath(a.out/'actor_mean_actions.npz'),**arrays)
    summary={}
    for method in ('TARGET-64','ORACLE-W1','ORACLE-ONLY'):
        selected=[x for x in rows if x['method']==method]
        summary[method]={k:dict(mean=statistics.mean(x[k] for x in selected),sample_sd=statistics.stdev(x[k] for x in selected)) for k in ('target_action_mse','source_neighbor_action_mse')}
    assert not torch.cuda.is_initialized()
    write(a.out/'completed.json',dict(passed=True,rows=rows,summary=summary,elapsed=time.time()-started,
      normalization='target state float64 mean and population SD + 1e-6',k=5,exact_knn=True,exhaustive_checks=12,
      metric='MSE across 5000 target states and 6 action coordinates; same target states for all actors',
      interpretation='Descriptive action proximity, not return or causal attribution; nearest source actions are proxies.',cuda_initialized=False))

if __name__=='__main__':main()
