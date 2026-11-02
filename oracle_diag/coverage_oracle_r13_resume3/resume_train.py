# -*- coding: utf-8 -*-
"""R203 full-state100k continuation only; no post-checkpoint old GPU overlap exists."""
from runtime import *
import argparse, random, runpy, traceback
import numpy as np
import torch
from learner import initialize, Sampler
from resume_state import restore_training, advance_sampler, tensor_equal
from tensorboardX import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
evaluate = runpy.run_path(str(HERE / 'train.py'))['evaluate']

def run(job_path):
    job=read(job_path); check_job(job)
    assert (job['arm'],job['seed'])==('R',203)
    accepted=read(OUT/'cpu_acceptance.json')
    assert accepted['CPU_only'] and accepted['CPU_split_resume_bitwise_identical']
    rec=read(OUT/'recovery.json')
    assert sha(rec['checkpoint'])==rec['checkpoint_sha256']
    assert not rec['old_GPU_loss_overlap_available']
    original=Path(rec['original_run'])
    configure_cpu(); assert torch.cuda.is_available()
    assert torch.cuda.get_device_name(0)==rec['expected_GPU']
    out=Path(job['out']); fresh(out); start=time.time()
    write(out/'started.json',dict(pid=os.getpid(),time=start,job_path=str(job_path),resume_step=100000))
    try:
        config=read(original/'config.json')
        assert config['torch']==torch.__version__ and config['python']==sys.version
        assert config['learner']==model_config(10.,500000,203)
        config.update(job=job,job_sha256=sha(job_path),code_sha256=code_hashes(),protocol_sha256=sha(OUT/'protocol.json'),
            resume_step=100000,resume_checkpoint=rec['checkpoint'],resume_checkpoint_sha256=rec['checkpoint_sha256'],
            original_config_path=str(original/'config.json'),original_config_sha256=sha(original/'config.json'))
        write(out/'config.json',config)
        model=initialize(config['learner'],'cuda:0')
        assert model.digest()==read(original/'initialization.json')['sha256']
        write(out/'initialization.json',read(original/'initialization.json'))
        saved=restore_training(model,rec['checkpoint'])
        assert model.total_it==100000 and model.finite()
        assert tensor_equal(saved,model.checkpoint()), 'GPU full-state restoration differs from original checkpoint'
        tar=target_data(); raw,oracle,raw_ids,meta=source_data()
        mask_record=read(OUT/'R_mask_receipts.json')['masks']['203']
        assert sha(mask_record['path'])==mask_record['sha256']
        ids=np.load(mask_record['path']); np.save(out/'R_mask.npy',ids)
        target=Sampler(tar,203,'target_sample','cuda:0')
        source=Sampler(oracle,203,'source_sample','cuda:0',ids)
        advance_sampler(target,100000); advance_sampler(source,100000)
        for name,sample in (('target_sample',target),('source_sample',source)):
            assert sample.hash.hexdigest()==accepted['stream_hashes'][name]['100000']
        alignment=read(original/'source_alignment.json')
        assert alignment['mask_sha256']==hashlib.sha256(source.ids.astype('<i8').tobytes()).hexdigest()
        assert alignment['receipt_sha256']==sha(ROOT/'oracle_diag/results/v3_20261008/source_oracle1/completed.json')
        write(out/'source_alignment.json',alignment)
        write(out/'GPU_STATE_RESTORED.json',dict(passed=True,checkpoint_step=100000,
            checkpoint_sha256=rec['checkpoint_sha256'],model_digest=model.digest(),
            policy_V_Q_targetQ_three_Adam_cosine_counter_exact=True,
            target_draws=target.draws,source_draws=source.draws,
            target_stream_sha256=target.hash.hexdigest(),source_stream_sha256=source.hash.hexdigest(),
            both_sampling_streams_match_old100k=True,sealed_R_mask_matches=True,
            GPU_loss_overlap_available=False,GPU_loss_overlap_verified=False,
            CPU_continuous_resumed_bitwise_test_passed=True,new_GPU_probe_gradient_updates=0,
            scheduler=model.policy_lr_schedule.state_dict(),time=time.time()))
        e=env(); writer=SummaryWriter(str(Path(shortpath(out))/'tb'))
        evaluations=[v for v in read(original/'evaluations.json')['events'] if v['step']<=100000]
        assert [v['step'] for v in evaluations]==list(range(10000,100001,10000))
        ea=EventAccumulator(str(Path(shortpath(original))/'tb'),size_guidance={'scalars':0}).Reload()
        prefix_count=0
        for tag in ('loss/V','loss/Q','loss/actor','evaluation/target_score'):
            for ev in ea.Scalars(tag):
                assert ev.step<=100000
                writer.add_scalar(tag,ev.value,ev.step,walltime=ev.wall_time); prefix_count+=1
        writer.flush()
        write(out/'prefix_provenance.json',dict(original_run=str(original),inherited_through_step=100000,
            original_evaluations_sha256=sha(original/'evaluations.json'),prefix_scalar_count=prefix_count,
            original_TensorBoard_files={str(f):sha(f) for f in (original/'tb').iterdir() if f.is_file()},
            no_post_checkpoint_old_GPU_loss_window=True,no_recomputed_overlap_metrics=True))
        write(out/'evaluations.json',dict(events=evaluations))
        eval_steps=sum(v['steps'] for v in evaluations)
        prior_duration=read(original/'progress.json')['elapsed']
        for step in range(100001,500001):
            losses=model.step(target.sample(),source.sample())
            assert model.total_it==step
            if step%1000==0:
                assert model.finite()
                for tag,val in zip(('loss/V','loss/Q','loss/actor'),losses): writer.add_scalar(tag,float(val),step)
                write(out/'progress.json',dict(step=step,budget=500000,resume_from=100000,new_updates=step-100000,
                    elapsed=time.time()-start,new_attempt_elapsed=time.time()-start,original_attempt_elapsed=prior_duration,
                    pid=os.getpid(),target_draws=target.draws,source_draws=source.draws,
                    target_stream_sha256=target.hash.hexdigest(),source_stream_sha256=source.hash.hexdigest(),
                    evaluation_steps=eval_steps,finite=True,stage='confirmation',GPU_state_restoration_verified=True,
                    GPU_loss_overlap_available=False),replace=True)
            if step%10000==0:
                result=evaluate(model,e,'confirmation',203); result['step']=step
                evaluations.append(result); eval_steps+=result['steps']
                writer.add_scalar('evaluation/target_score',result['mean'],step); writer.flush()
                write(out/'evaluations.json',dict(events=evaluations),replace=True)
            if step%100000==0:
                checkpoint=model.checkpoint()
                checkpoint['recovery_random_state']=dict(target_rng=target.rng.get_state(),source_rng=source.rng.get_state(),
                    target_draws=target.draws,source_draws=source.draws,
                    target_stream_sha256=target.hash.hexdigest(),source_stream_sha256=source.hash.hexdigest(),
                    torch_rng=torch.get_rng_state(),torch_cuda_rng=torch.cuda.get_rng_state_all(),numpy_rng=np.random.get_state(),python_rng=random.getstate())
                torch.save(checkpoint,str(out/('model_%d.pt'%step)))
        writer.close(); e.close()
        final=out/'model_500000.pt'
        write(out/'sampling.json',dict(target_first=target.first,source_first=source.first,
            target_draws=target.draws,source_draws=source.draws,
            target_stream_sha256=target.hash.hexdigest(),source_stream_sha256=source.hash.hexdigest()))
        write(out/'early_losses.json',dict(trace=[],original_first10_in_memory_trace_unavailable=True,
            GPU_loss_overlap_available=False,GPU_exact_full_state_restoration_receipt=str(out/'GPU_STATE_RESTORED.json')))
        write(out/'completed.json',dict(job=job,steps=500000,final_model=str(final),final_model_sha256=sha(final),
            finite=model.finite(),model_digest=model.digest(),elapsed=time.time()-start,new_attempt_elapsed=time.time()-start,
            original_attempt_elapsed=prior_duration,elapsed_excludes_original_prefix=True,evaluation_steps=eval_steps,ended=time.time(),
            resumed_from=100000,new_updates=400000,source_attempt=str(original),GPU_state_restoration_exact=True,GPU_loss_overlap_available=False))
    except BaseException:
        write(out/'failed.json',dict(pid=os.getpid(),time=time.time(),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--job',type=Path,required=True)
    run(p.parse_args().job)
