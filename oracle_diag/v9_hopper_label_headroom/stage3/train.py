from runtime import *
import argparse,random,traceback
import numpy as np
import torch
from learner import Actor,Sampler
from tensorboardX import SummaryWriter
def evaluate(model, e, job):
    saved=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    returns,lengths,seeds=[],[],[]
    try:
        for episode in range(10):
            value=500000+10*(job['seed']-301)+episode
            seeds.append(value); e.seed(value)
            state,done,ret,length=e.reset(),False,0.,0
            while not done:
                state,reward,done,_=e.step(model.deterministic_action(state))
                ret+=float(reward); length+=1
            returns.append(ret); lengths.append(length)
    finally:
        random.setstate(saved[0]); np.random.set_state(saved[1])
        torch.set_rng_state(saved[2]); torch.cuda.set_rng_state_all(saved[3])
    spec=task_spec(job['task'])
    scores=(np.asarray(returns)-spec['score_min'])/(spec['score_max']-spec['score_min'])*100.
    expected='Walker2dEnv' if job['task']=='walker2d-morph' else 'HopperEnv'
    assert type(e.unwrapped).__name__==expected and e._max_episode_steps==1000
    assert abs(float(e.unwrapped.model.opt.timestep)-.002)<1e-12
    return dict(raw_returns=returns,scores=scores.tolist(),mean=float(scores.mean()),episode_lengths=lengths,
        env_seeds=seeds,steps=sum(lengths),actual_env_class=type(e.unwrapped).__module__,
        actual_shifted_object=True,xml_sha256=spec['xml_sha256'],timestep=float(e.unwrapped.model.opt.timestep),
        normalization_min=spec['score_min'],normalization_max=spec['score_max'],preserve_training_RNG=True)

def run(job_path):
    job=read(job_path);validate(job);configure_cpu();assert torch.cuda.is_available()
    assert read(STAGE2/'STAGE_COMPLETED.json')['passed'] and read(STAGE2/'DELIVERY_VERIFIED_STAGE2.json')['passed']
    out=Path(job['out']);fresh(out);write(out/'started.json',dict(pid=os.getpid(),time=time.time(),job_path=str(job_path)))
    write(out/'config.json',dict(job=job,policy_class='offline_offline.iql_unc.Policy (R14 IQL_DYN actor)',actor_lr=3e-4,hidden=256,batch=64,beta=0,advantage_weight=1,
        policy_loss='IQL_DYN beta0: mean_batch(mean_action(Policy.bc_loss(state,action)))',Q_or_V_constructed=False,Q_or_V_optimized=False,source_rows=0,
        cosine_T_max=200000,observation_scaling='raw',reward_scaling=1,protocol_sha256=sha(OUT/'protocol.json'),code_sha256=code_hashes()))
    try:
        arrays=target_data(job['task']);model=Actor(job,'cuda:0');sample=Sampler(arrays,job,'cuda:0')
        write(out/'initialization.json',dict(sha256=model.digest()));e=env(job['task']);writer=SummaryWriter(str(Path(shortpath(out))/'tb'))
        start=time.time();events=[];envsteps=0
        for step in range(1,200001):
            loss=model.step(*sample.sample())
            if step%1000==0:
                assert model.finite() and bool(torch.isfinite(loss))
                write(out/'progress.json',dict(step=step,budget=200000,pid=os.getpid(),elapsed=time.time()-start,finite=True,draws=sample.draws,sampling_sha256=sample.hash.hexdigest(),evaluation_steps=envsteps),replace=True)
                writer.add_scalar('loss/actor',float(loss),step)
            if step%5000==0:
                v=evaluate(model,e,job);v['step']=step;events.append(v);envsteps+=v['steps']
                writer.add_scalar('evaluation/target_score',v['mean'],step);writer.flush();write(out/'evaluations.json',dict(events=events),replace=True)
            if step%50000==0:
                path=out/('model_%d.pt'%step);torch.save(model.checkpoint(),winpath(path))
        writer.close();e.close();path=out/'model_200000.pt'
        write(out/'sampling.json',dict(draws=sample.draws,sampling_sha256=sample.hash.hexdigest(),selected_rows=len(sample.ids)))
        write(out/'completed.json',dict(job=job,steps=200000,finite=model.finite(),final_model=str(path),final_model_sha256=sha(path),model_digest=model.digest(),elapsed=time.time()-start,evaluation_steps=envsteps,ended=time.time()))
    except BaseException:
        write(out/'failed.json',dict(pid=os.getpid(),time=time.time(),traceback=traceback.format_exc()));raise
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',type=Path,required=True);run(p.parse_args().job)
