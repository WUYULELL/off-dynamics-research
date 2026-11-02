"""One authorized raw-source fixed-weight run, preserving R13 update/evaluation order."""
from runtime import *
import argparse
import random
import traceback
import numpy as np
import torch
from learner import initialize, Sampler
from tensorboardX import SummaryWriter

def evaluate(model, e, job):
    saved=(random.getstate(),np.random.get_state(),torch.get_rng_state(),torch.cuda.get_rng_state_all())
    returns,lengths,seeds=[],[],[]
    try:
        for episode in range(10):
            value=500000+10*(job['seed']-201)+episode
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
    job=read(job_path); check_job(job); configure_cpu()
    assert torch.cuda.is_available(),'No CPU fallback for long training'
    out=Path(job['out']); fresh(out)
    write(out/'started.json',dict(pid=os.getpid(),time=time.time(),job_path=str(job_path)))
    config=model_config(job['temp'],job['steps'],job['seed'],job['task'],job['source_w'])
    write(out/'config.json',dict(job=job,learner=config,protocol_sha256=sha(OUT/'protocol.json'),
        job_sha256=sha(job_path),code_sha256=code_hashes(),python=sys.version,torch=torch.__version__,
        gpu=torch.cuda.get_device_name(0),learner_observations='raw',reward_scale=1.,
        source_coefficient=job['source_w'],source_lambda=job['source_lambda'],source_labels=job['source_labels'],
        V_source=job['arm']=='M',Q_source=job['arm']=='M',actor_source=False,batch_target=64,
        batch_source=64 if job['arm']=='M' else 0,actor_advantage='pre-V-step target advantage',
        target_Q_update='every gradient update',source_sampling='full canonical source uniformly with replacement',
        random_seeds={k:seed(k,job['seed'],job['task']) for k in ('initialization','target_sample','source_sample')}))
    try:
        target_arrays=target_data(job['task'])
        model=initialize(config,'cuda:0')
        write(out/'initialization.json',dict(sha256=model.digest()))
        target=Sampler(target_arrays,job['seed'],'target_sample','cuda:0',task=job['task'])
        source=None
        if job['arm']=='M':
            arrays,raw_ids=source_data(job['task'])
            source=Sampler(arrays,job['seed'],'source_sample','cuda:0',task=job['task'])
            write(out/'source_alignment.json',dict(n=len(raw_ids),selected=len(raw_ids),
                source_labels='original_dataset',oracle_loaded=False,source_dataset=task_spec(job['task'])['source_dataset'],
                canonical_raw_ids_sha256=hashlib.sha256(raw_ids.astype('<i8').tobytes()).hexdigest(),
                array_sha256={name:hashlib.sha256(a.tobytes()).hexdigest() for name,a in zip(('state','action','next_state','reward','not_done'),arrays)}))
        e=env(job['task']); writer=SummaryWriter(str(Path(shortpath(out))/'tb'))
        evaluations,trace=[],[]; start=time.time();eval_steps=0
        for step in range(1,job['steps']+1):
            losses=model.step(target.sample(),None if source is None else source.sample())
            if step<=10: trace.append(dict(step=step,V=float(losses[0]),Q=float(losses[1]),actor=float(losses[2])))
            if step%1000==0:
                assert model.finite(),'Non-finite model'
                write(out/'progress.json',dict(step=step,budget=job['steps'],elapsed=time.time()-start,pid=os.getpid(),
                    target_draws=target.draws,source_draws=0 if source is None else source.draws,
                    target_stream_sha256=target.hash.hexdigest(),source_stream_sha256=None if source is None else source.hash.hexdigest(),
                    evaluation_steps=eval_steps,finite=True,stage=job['phase']),replace=True)
                for k,v in zip(('V','Q','actor'),losses):writer.add_scalar('loss/'+k,float(v),step)
            if step%10000==0:
                result=evaluate(model,e,job);result['step']=step;evaluations.append(result);eval_steps+=result['steps']
                writer.add_scalar('evaluation/target_score',result['mean'],step);writer.flush()
                write(out/'evaluations.json',dict(events=evaluations),replace=True)
            if step%100000==0:torch.save(model.checkpoint(),str(out/('model_%d.pt'%step)))
        writer.close();e.close();final_path=out/('model_%d.pt'%job['steps'])
        write(out/'sampling.json',dict(target_first=target.first,source_first=None if source is None else source.first,
            target_draws=target.draws,source_draws=0 if source is None else source.draws,
            target_stream_sha256=target.hash.hexdigest(),source_stream_sha256=None if source is None else source.hash.hexdigest()))
        write(out/'early_losses.json',dict(trace=trace))
        write(out/'completed.json',dict(job=job,steps=job['steps'],final_model=str(final_path),final_model_sha256=sha(final_path),
            finite=model.finite(),model_digest=model.digest(),elapsed=time.time()-start,evaluation_steps=eval_steps,ended=time.time()))
    except BaseException:
        write(out/'failed.json',dict(pid=os.getpid(),time=time.time(),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--job',type=Path,required=True)
    run(parser.parse_args().job)
