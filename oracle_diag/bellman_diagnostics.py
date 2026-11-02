"""Stage 2b for a saved fixed V; original kNN function is called directly."""
from verify_replay_v2 import *
from prepare_diagnostics import corr
import argparse


def weights(b):
    raw=np.exp(-b/(np.mean(b)+1e-6))
    return np.clip(.2*raw/(np.mean(raw)+1e-6),.05,1.)


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',required=True);p.add_argument('--kind',required=True)
    p.add_argument('--checkpoint',type=Path,required=True);args=p.parse_args()
    folder=OUT/'diagnostics';output=folder/(args.env+'_'+args.kind+'_bellman.json')
    if output.exists():raise RuntimeError('Refusing overwrite')
    src=np.load(folder/(args.env+'_source_oracle.npz'));geo=np.load(folder/(args.env+'_geometry.npz'))
    from common import load_policy_config
    from offline_offline.iql_bellman_fb import IQL_BELLMAN_FB
    cfg=load_policy_config(args.env,overrides=dict(source_budget=.2,bellman_eta=1.,seed=0,max_step=500000))
    model=IQL_BELLMAN_FB(cfg,torch.device('cpu'))
    model.v_func.load_state_dict(torch.load(str(args.checkpoint),map_location='cpu'));model.v_func.eval()
    tensor=lambda x:torch.as_tensor(x,dtype=torch.float32)
    for attribute,key in [('target_state_bank','bank_state'),('target_next_state_bank','bank_next'),
                          ('target_reward_bank','bank_reward'),('target_not_done_bank','bank_not_done')]:
        t=tensor(geo[key]);setattr(model,attribute,t.reshape(-1,1) if t.ndim==1 else t)
    # Geometry snapshot was produced by the original initialize_target_feature_bank.
    model.target_mean=tensor(geo['bank_mean']);model.target_std=tensor(geo['bank_std'])
    # Actions are absent from the snapshot; reload the original bank once and assert identity.
    tar,full=load_target_buffer(args.env,0)
    model.initialize_target_feature_bank(tar)
    assert np.array_equal(model.target_state_bank.numpy(),geo['bank_state'])
    assert np.array_equal(model.target_std.numpy(),geo['bank_std'])
    ids=geo['query_indices'];j=geo['knn_indices'];N=len(ids);values=[];actual_weights=[]
    with torch.no_grad():
        value=lambda x:model.v_func(tensor(x)).numpy().ravel()
        ys=(tensor(src['source_reward'][ids])+.99*tensor(src['source_not_done'][ids])*model.v_func(tensor(src['source_next'][ids])).ravel()).numpy()
        yo=(tensor(src['oracle_reward'][ids])+.99*tensor(src['oracle_not_done'][ids])*model.v_func(tensor(src['oracle_next'][ids])).ravel()).numpy()
        real=np.abs(ys-yo)
        for start in range(0,N,128):
            rows=ids[start:start+128]
            stats=model.compute_source_weight(tensor(src['source_state'][rows]),tensor(src['source_action'][rows]),
                tensor(src['source_next'][rows]),tensor(src['source_reward'][rows]).reshape(-1,1),
                tensor(src['source_not_done'][rows]).reshape(-1,1))
            values.append(stats['bellman_mismatch'].numpy().ravel());actual_weights.append(stats['weight'].numpy().ravel())
        knn=np.concatenate(values)
        # B's point estimate: V at corrected mean next state, mean neighbor reward/mask.
        diff_ref=geo['bank_reward'][j].mean(1)+.99*geo['bank_not_done'][j].mean(1)*value(geo['pred_b'])
        diff=np.abs(ys-diff_ref)
        shifted=src['source_state'][ids,None,:]+geo['bank_next'][j]-geo['bank_state'][j]
        sens_ref=(geo['bank_reward'][j]+.99*geo['bank_not_done'][j]*value(shifted.reshape(-1,shifted.shape[-1])).reshape(N,5)).mean(1)
        sensitivity=np.abs(ys-sens_ref)
    clean=geo['clean'];groups=N//128;ret={}
    top=lambda x:np.lexsort((ids,-x))[:int(np.ceil(.2*N))]
    for name,estimated in [('knn',knn),('diff_mean_state',diff),('diff_per_neighbor_sensitivity',sensitivity),('oracle',real)]:
        ranks=[];ess=[];means=[]
        for start in range(0,groups*128,128):
            b=estimated[start:start+128];truth=real[start:start+128];r=corr(b,truth)
            if r is not None:ranks.append(r)
            w=weights(b);ess.append(float(w.sum()**2/(np.square(w).sum()+1e-6)/128));means.append(float(w.mean()))
        ret[name]=dict(global_spearman=corr(estimated,real),clean_global_spearman=corr(estimated[clean],real[clean]),
            batch_spearman_mean=float(np.mean(ranks)) if ranks else None,batch_spearman_distribution=summarize(ranks),
            full_batches=groups,remainder_excluded_from_batch_metrics=N-groups*128,
            top20_overlap=float(len(set(top(estimated))&set(top(real)))/len(top(real))),top20_n=len(top(real)),
            ess_ratio=summarize(ess),source_weight_mean=summarize(means),mismatch=summarize(estimated))
    actual=np.concatenate(actual_weights)
    expected=np.concatenate([weights(knn[start:start+128]) for start in range(0,N,128)])
    assert np.allclose(actual,expected,atol=3e-6,rtol=3e-6)
    data=dict(env=args.env,V=args.kind,checkpoint=str(args.checkpoint),checkpoint_sha256=sha256_file(args.checkpoint),
        n=N,seed=0,metrics=ret,diff_definition='mean neighbor reward + gamma * mean neighbor not_done * V(query + mean neighbor displacement)',
        note='Fixed learned V, not calibrated truth. Oracle simulator validation is approximate. Hopper true physical shift is zero on this source support; tiny residual/noise ranks are not dynamics evidence.',
        direct_knn_weight_formula_verified=True)
    output.write_text(json.dumps(data,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    np.savez_compressed(folder/(args.env+'_'+args.kind+'_bellman.npz'),ids=ids,oracle=real,knn=knn,diff=diff,
        diff_per_neighbor=sensitivity,clean=clean,source_y=ys,oracle_y=yo)
    print(json.dumps(data,ensure_ascii=False),flush=True)


if __name__=='__main__':main()
