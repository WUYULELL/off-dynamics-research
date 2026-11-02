"""Continue only the interrupted T101 trial, in a new directory, at step400001."""
from runtime import *
import argparse
import random
import traceback
import numpy as np
import torch
from learner import initialize,Sampler
from resume_state import restore_training,advance_sampler
from train import evaluate
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

def run(job_path):
    job=read(job_path)
    check_job(job)
    assert (job['arm'],job['phase'],job['seed'],job['temp'],job['steps'])==('T','tune',101,10.,1000000)
    recovery=protocol()['recovery']
    assert sha(recovery['checkpoint'])==recovery['checkpoint_sha256']
    assert read(OUT/'cpu_acceptance.json')['CPU_split_resume_bitwise_identical']
    configure_cpu()
    assert torch.cuda.is_available(),'No silent CPU fallback'
    out=Path(job['out']);fresh(out)
    original=Path(recovery['checkpoint']).parent
    start=time.time()
    write(out/'started.json',dict(pid=os.getpid(),time=start,job_path=str(job_path),resume_step=400000))
    try:
        config=read(original/'config.json')
        assert config['learner']==model_config(10.,1000000,101)
        config.update(job=job,protocol_sha256=sha(OUT/'protocol.json'),job_sha256=sha(job_path),
            code_sha256=code_hashes(),resume_checkpoint=recovery['checkpoint'],
            resume_checkpoint_sha256=recovery['checkpoint_sha256'],resume_step=400000,
            original_config_path=str(original/'config.json'),original_config_sha256=sha(original/'config.json'))
        write(out/'config.json',config)
        model=initialize(config['learner'],'cuda:0')
        assert model.digest()==read(original/'initialization.json')['sha256']
        write(out/'initialization.json',read(original/'initialization.json'))
        restore_training(model,recovery['checkpoint'])
        target=Sampler(target_data(),101,'target_sample','cuda:0')
        advance_sampler(target,400000)
        assert target.hash.hexdigest()==read(OUT/'cpu_acceptance.json')['stream_hashes']['400000']
        write(out/'resume_state.json',dict(step=model.total_it,model_digest=model.digest(),
            checkpoint_sha256=recovery['checkpoint_sha256'],target_draws=target.draws,
            target_stream_sha256=target.hash.hexdigest(),scheduler=model.policy_lr_schedule.state_dict(),
            weights_targetQ_optimizers_scheduler_counter_restored=True))
        e=env()
        writer=SummaryWriter(str(Path(shortpath(out))/'tb'))
        source_evals=read(original/'evaluations.json')['events']
        evaluations=[x for x in source_evals if x['step']<=400000]
        assert [x['step'] for x in evaluations]==list(range(10000,400001,10000))
        ea=EventAccumulator(str(Path(shortpath(original))/'tb'),size_guidance={'scalars':0}).Reload()
        prefix_count=0
        for tag in ('loss/V','loss/Q','loss/actor','evaluation/target_score'):
            for event in ea.Scalars(tag):
                if event.step<=400000:
                    writer.add_scalar(tag,event.value,event.step,walltime=event.wall_time)
                    prefix_count+=1
        writer.flush()
        write(out/'prefix_provenance.json',dict(original_run=str(original),inherited_through_step=400000,
            original_evaluations_sha256=sha(original/'evaluations.json'),prefix_scalar_count=prefix_count,
            excluded_old_steps_after400k=True,original_TensorBoard_files={str(f):sha(f) for f in (original/'tb').iterdir() if f.is_file()}))
        write(out/'evaluations.json',dict(events=evaluations))
        overlap=read(OUT/'OVERLAP_LOSSES.json')['events'];checked=[]
        eval_steps=sum(x['steps'] for x in evaluations)
        previous_elapsed=read(original/'progress.json')['elapsed']
        for step in range(400001,1000001):
            losses=model.step(target.sample())
            assert model.total_it==step
            if step%1000==0:
                assert model.finite()
                for tag,val in zip(('loss/V','loss/Q','loss/actor'),losses):
                    value=float(val)
                    if str(step) in overlap[tag]:
                        expected=overlap[tag][str(step)]
                        checked.append(dict(tag=tag,step=step,expected=expected,actual=value,exact=value==expected))
                        if value!=expected:
                            write(out/'overlap_failed.json',dict(checks=checked,rule='exact float32; no tolerance relaxation'))
                            raise RuntimeError('Resumed overlap differs at %s/%d; stop without changing tolerance'%(tag,step))
                    writer.add_scalar(tag,value,step)
                if step==406000:
                    assert len(checked)==18 and all(x['exact'] for x in checked)
                    write(out/'OVERLAP_VERIFIED.json',dict(passed=True,checks=checked,
                        exact_float32=True,no_extra_steps_beyond_resumed_budget=True))
                write(out/'progress.json',dict(step=step,budget=1000000,resume_from=400000,
                    new_updates=step-400000,elapsed=previous_elapsed+time.time()-start,
                    new_attempt_elapsed=time.time()-start,pid=os.getpid(),target_draws=target.draws,source_draws=0,
                    target_stream_sha256=target.hash.hexdigest(),source_stream_sha256=None,
                    evaluation_steps=eval_steps,finite=True,stage='tune',
                    GPU_overlap_verified=step>=406000),replace=True)
            if step%10000==0:
                result=evaluate(model,e,'tune',101);result['step']=step
                evaluations.append(result);eval_steps+=result['steps']
                writer.add_scalar('evaluation/target_score',result['mean'],step);writer.flush()
                write(out/'evaluations.json',dict(events=evaluations),replace=True)
            if step%100000==0:
                checkpoint=model.checkpoint()
                checkpoint['recovery_random_state']=dict(target_rng=target.rng.get_state(),target_draws=target.draws,
                    target_stream_sha256=target.hash.hexdigest(),torch_rng=torch.get_rng_state(),
                    torch_cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
                torch.save(checkpoint,str(out/('model_%d.pt'%step)))
        writer.close();e.close()
        assert read(out/'OVERLAP_VERIFIED.json')['passed']
        final=out/'model_1000000.pt'
        write(out/'sampling.json',dict(target_first=target.first,source_first=None,target_draws=target.draws,
            source_draws=0,target_stream_sha256=target.hash.hexdigest(),source_stream_sha256=None))
        write(out/'early_losses.json',dict(trace=[],limitation='original first10 loss trace was held in memory, unavailable after crash',
            overlap_trace_available=str(out/'OVERLAP_VERIFIED.json')))
        write(out/'completed.json',dict(job=job,steps=1000000,final_model=str(final),final_model_sha256=sha(final),
            finite=model.finite(),model_digest=model.digest(),elapsed=previous_elapsed+time.time()-start,
            new_attempt_elapsed=time.time()-start,evaluation_steps=eval_steps,ended=time.time(),
            resumed_from=400000,new_updates=600000,source_attempt=str(original),GPU_overlap_exact=True))
    except BaseException:
        write(out/'failed.json',dict(pid=os.getpid(),time=time.time(),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--job',required=True,type=Path);run(p.parse_args().job)
