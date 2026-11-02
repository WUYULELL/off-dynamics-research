"""Stage 1 aligned oracle arrays, Stage 2a original-bank geometry (CPU)."""
from verify_replay_v2 import *
from scipy.stats import spearmanr


def corr(a,b):
    if np.ptp(a)==0 or np.ptp(b)==0:return None
    v=float(spearmanr(a,b).correlation)
    return v if np.isfinite(v) else None


def main():
    gate=json.loads((OUT/'final_acceptance.json').read_text())
    if not gate['all_passed']:raise RuntimeError('Replay gate not passed')
    destination=OUT/'diagnostics'
    if destination.exists():raise RuntimeError('Refusing overwrite')
    destination.mkdir()
    summary={}
    for name in ENVS:
        print('PREPARE',name,flush=True)
        src=load_source_buffer(name);tar,full=load_target_buffer(name,0);agent=build_bellman_agent(name,0,tar)
        states=np.load(OUT/(name+'_validated_states.npz'))
        assert len(states['target_reward'])==src.size
        std=agent.target_std[0,:src.state.shape[1]].numpy().astype(float)
        arr=dict(indices=np.arange(src.size),source_state=src.state,source_action=src.action,
            source_next=src.next_state,source_reward=src.reward.ravel(),source_not_done=src.not_done.ravel(),
            oracle_next=states['target_next'],oracle_reward=states['target_reward'],oracle_not_done=1.-states['target_done'],
            source_physical_next=states['source_next'],source_physical_reward=states['source_reward'],
            boundary=states['boundary'],clipped=states['clipped'],target_state_std=std)
        np.savez_compressed(destination/(name+'_source_oracle.npz'),**arr)
        ids=np.random.default_rng(0).permutation(src.size)[:min(20000,src.size)]
        distances=[];indices=[]
        with torch.no_grad():
            for start in range(0,len(ids),128):
                rows=ids[start:start+128]
                f=agent.build_sa_features(torch.tensor(src.state[rows]),torch.tensor(src.action[rows]))
                d,j=agent.compute_knn_indices((f-agent.target_mean)/agent.target_std)
                distances.append(d.numpy());indices.append(j.numpy())
        j=np.concatenate(indices);d=np.concatenate(distances)
        bank_next=agent.target_next_state_bank.numpy();bank_state=agent.target_state_bank.numpy()
        a=bank_next[j].mean(1);b=src.state[ids]+(bank_next[j]-bank_state[j]).mean(1);c=src.next_state[ids]
        truth=states['target_next'][ids];real=np.linalg.norm((c-truth)/std,axis=1)
        clean=~states['boundary'][ids]
        metrics={}
        for label,pred in [('A_knn_next_mean',a),('B_displacement',b),('C_no_shift',c)]:
            error=np.linalg.norm((pred-truth)/std,axis=1)
            estimated=np.linalg.norm((c-pred)/std,axis=1)
            metrics[label]=dict(error=summarize(error),clean_error=summarize(error[clean]),
                estimated_vs_true_spearman=corr(estimated,real),
                clean_estimated_vs_true_spearman=corr(estimated[clean],real[clean]),
                neighbor_distance_vs_error_spearman=corr(d.mean(1),error))
        np.savez_compressed(destination/(name+'_geometry.npz'),query_indices=ids,knn_indices=j,knn_distance=d,
            pred_a=a,pred_b=b,pred_c=c,oracle=truth,real_mismatch=real,clean=clean,
            bank_state=bank_state,bank_next=bank_next,bank_reward=agent.target_reward_bank.numpy().ravel(),
            bank_not_done=agent.target_not_done_bank.numpy().ravel(),bank_mean=agent.target_mean.numpy(),bank_std=agent.target_std.numpy())
        summary[name]=dict(seed=0,n_queries=len(ids),requested_queries=20000,sampling='All 4999 unique rows in seed-0 shuffled order; no duplication',
            source_sha256=sha256_file(source_hdf5_path(name)),oracle_sha256=sha256_file(destination/(name+'_source_oracle.npz')),
            alignment='row i equals call_tar_dataset -> ReplayBuffer.convert_D4RL row i; raw hdf5 row i for i<N-1',
            target_subset_size=tar.size,target_bank_size=len(bank_state),knn_k=5,
            standardized_dynamics_mismatch=summarize(real),clean_standardized_dynamics_mismatch=summarize(real[clean]),
            physical_shift_standardized=summarize(np.linalg.norm((states['source_next']-states['target_next'])/std,axis=1)),
            reward_difference=summarize(np.abs(src.reward.ravel()-states['target_reward'])),
            boundary_rows=np.flatnonzero(states['boundary']).tolist(),stage2a=metrics)
        print(json.dumps(summary[name],ensure_ascii=False),flush=True)
    (destination/'stage1_stage2a.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False),encoding='utf-8')


if __name__=='__main__':main()
