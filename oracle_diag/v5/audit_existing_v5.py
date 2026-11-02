"""Audit reused baseline scalar metrics now; GPU reuse equivalence remains a separate gate."""
from audit_v5 import *

def main():
    rows=[];protocol=read(OUT/'protocol.json')
    for method,prefix,base in [('TARGET-64','h',V4_OUT),('ORACLE-W1','w',V4_OUT),('TARGET','t',ROOT/'oracle_diag/results/v3_20261008')]:
        for seed in range(5):
            folder=base/('b1/'+prefix+str(seed));assert read(folder/'completed.json')['steps']==500000
            f,sc=events(folder);scores=sc['test/target_normalized_score'];raw=sc['test/target_return']
            assert sorted(scores)==list(range(10000,500001,10000))
            assert all(abs(scores[k]-normalized(raw[k]))<2e-5 for k in scores)
            ev=read(folder/'evaluation_verified.json')['events']
            assert all(r['confirmed_target_env_object'] and r['env_class']=='gym.envs.mujoco.walker2d_v3' for r in ev)
            if method!='TARGET':assert all(r['time_limit']==1000 for r in ev)
            last=[scores[k] for k in protocol['A']['last5_steps']]
            rows.append(dict(stack='A',method=method,seed=seed,out=str(folder),tb_file=str(f),tb_sha256=sha(f),scores=scores,raw=raw,last5=last,last5_mean=statistics.mean(last),final_score=scores[500000]))
    study=read(ROOT/'reproductions/otdf_setup/study_five_seeds.json')
    for key in ('inputs_manifest','cost_manifest'):assert sha(study[key]['path'])==study[key]['sha256']
    expected=[46.8647,38.5873,40.3669,46.2825,46.5429]
    for old in study['seeds']:
        f,sc=events(old['run_dir']);raw=sc['test/target_return'];assert sorted(raw)==list(range(10000,1000001,10000))
        scores={step:normalized(value) for step,value in raw.items()};last=[scores[k] for k in protocol['B']['last5_steps']]
        mean=statistics.mean(last);assert abs(mean-expected[old['seed']-1])<.0001
        rows.append(dict(stack='B',method='OTDF',seed=old['seed'],out=old['run_dir'],tb_file=str(f),tb_sha256=sha(f),raw=raw,scores=scores,last5=last,last5_mean=mean,final_score=scores[1000000]))
    write(OUT/'existing_baselines.json',dict(passed=True,rows=rows,reuse_status='Original metrics verified; new GPU reuse equivalence not yet tested'))
    print('20 historical baseline runs independently read from original TensorBoard')

if __name__=='__main__':main()
