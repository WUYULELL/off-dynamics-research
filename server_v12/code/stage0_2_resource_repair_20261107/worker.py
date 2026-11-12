"""One fixed V12 0.2 Hopper T job. No resume, retry, or formal-training mode."""
import os
import sys

sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ.pop('ODRL_PAPER_STYLE_SOURCE_TARGET',None)

import argparse
import hashlib
import math
from pathlib import Path
import pickle
import random
import time
import traceback
from common import HERE,read,write,sha,utc,verify_package,private_binding,gpu_info,length_stats,loss_stats,hash_tree
from science import Science


def rng_snapshot(torch,np,sampler):
    return dict(Python=hashlib.sha256(pickle.dumps(random.getstate(),protocol=4)).hexdigest(),
        NumPy=hashlib.sha256(pickle.dumps(np.random.get_state(),protocol=4)).hexdigest(),
        Torch_CPU=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
        Torch_CUDA=[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in torch.cuda.get_rng_state_all()],
        sampler_rng=hashlib.sha256(pickle.dumps(sampler.rng.get_state(),protocol=4)).hexdigest(),
        sampler_draws=sampler.draws,sampler_sha256=sampler.hash.hexdigest())


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True)
    args=ap.parse_args();verify_package()
    assert sys.platform=='linux' and sys.version_info[:3]==(3,8,13)
    visible=os.environ.get('CUDA_VISIBLE_DEVICES','')
    assert visible.startswith('GPU-') and ',' not in visible, 'Exactly one explicit GPU UUID is required'
    job_path=args.job.resolve();job=read(job_path)
    root=Path(job['project']).resolve();base=job_path.parent.resolve()
    authorized=read(base/'WORKER_AUTHORIZATION.json')
    assert authorized['scope']=='V12_0.2_three_fixed_jobs_only'
    assert authorized['jobs'][job['id']]==sha(job_path)
    assert job['id'] in ('determinism_a','determinism_b','entry_20k')
    assert job['seed']==201 and job['task']=='hopper-morph'
    assert job['steps']==(20000 if job['id']=='entry_20k' else 10000)
    assert job['evaluate']==(job['id']=='entry_20k')
    assert job['gpu_uuid']==visible and job['max_step']==500000 and job['temp']==10
    out=base/job['id'];out.mkdir(exist_ok=False)
    counts=dict(learner_updates=0,Adam_updates=0,environment_construction_attempts=0,
                environment_instances=0,constructor_environment_steps=0,evaluation_environment_steps=0)
    model=None;env=None;inflight_step=None
    try:
        for name in ('temporary_files','cache','cuda_cache'):(out/name).mkdir()
        os.environ['TMPDIR']=str(out/'temporary_files');os.environ['XDG_CACHE_HOME']=str(out/'cache')
        os.environ['CYTHON_CACHE_DIR']=str(out/'cache/cython')
        os.environ['CUDA_CACHE_PATH']=str(out/'cuda_cache')
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        # Preserve PyTorch 1.11's recorded defaults; fixed before seeing any result.
        torch.backends.cuda.matmul.allow_tf32=True
        torch.backends.cudnn.allow_tf32=True
        science=Science(root);science.load()
        config=science.config()
        data=science.runtime.target_data('hopper-morph')
        assert all(a.dtype==np.float32 and np.isfinite(a).all() for a in data)
        assert torch.__version__=='1.11.0+cu113' and torch.version.cuda=='11.3'
        assert torch.cuda.is_available() and torch.cuda.device_count()==1
        assert 'A6000' in torch.cuda.get_device_name(0)
        physical=gpu_info(visible)
        write(out/'config.json',dict(job=job,learner=config,target_rows=len(data[0]),target_batch=64,
            source_batch=0,source_in_V_Q_actor=False,source_data_or_oracle_loaded=False,
            actor_loss='exp(10*pre_V_advantage).clamp(max=100) * action-dimension mean MSE; target batch mean',
            training_observations='raw',reward_scale=1.,evaluate_every=10000,episodes_per_evaluation=10,
            target_Q_update='every learner update',cosine_max_step=500000,
            Torch=torch.__version__,Torch_CUDA_runtime=torch.version.cuda,device_name=torch.cuda.get_device_name(0),
            gpu=physical,deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
            cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,
            matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,
            CUBLAS_WORKSPACE_CONFIG=os.environ['CUBLAS_WORKSPACE_CONFIG'],threads=1,
            scientific_AST=science.science,worker_sha256=sha(HERE/'worker.py')))
        model=science.learner['initialize'](config,'cuda:0')
        sampler=science.learner['Sampler'](data,201,'target_sample','cuda:0',task='hopper-morph')
        write(out/'initialization.json',dict(parameter_sha256=model.digest(),
            initialization_seed=science.runtime.seed('initialization',201,'hopper-morph'),
            target_sampling_seed=science.runtime.seed('target_sample',201,'hopper-morph')))
        write(out/'STARTED.json',dict(pid=os.getpid(),ppid=os.getppid(),time=utc(),gpu_uuid=visible))
        if job['evaluate']:
            private=private_binding(out)
            os.environ['MUJOCO_PY_FORCE_CPU']='1'
            import mujoco_py
            assert private in Path(mujoco_py.__file__).resolve().parents
            extension=Path(mujoco_py.cymj.__file__).resolve()
            assert private in extension.parents
            write(out/'NATIVE_RUNTIME.json',dict(extension=str(extension),extension_sha256=sha(extension),
                python_binding=str(Path(mujoco_py.__file__).resolve()),input_hashes=hash_tree(private)))
            import gym
            from gym.envs.mujoco.hopper_v3 import HopperEnv
            spec=science.task_spec('hopper-morph')
            counts['environment_construction_attempts']+=1
            raw=HopperEnv(xml_file=spec['xml'])
            counts['environment_instances']+=1;counts['constructor_environment_steps']+=1
            original_step=raw.step
            def counted_step(action):
                counts['evaluation_environment_steps']+=1
                return original_step(action)
            raw.step=counted_step
            env=gym.wrappers.TimeLimit(raw,max_episode_steps=1000)
        losses=np.empty((job['steps'],3),dtype=np.float32)
        evaluations=[];eval_rng=[];first10k=None
        torch.cuda.synchronize();started=time.monotonic()
        with (out/'losses.jsonl').open('x',encoding='utf-8') as log:
            for step in range(1,job['steps']+1):
                inflight_step=step
                values=model.step(sampler.sample(),None)
                counts['learner_updates']+=1;counts['Adam_updates']+=3
                inflight_step=None
                x=[float(v) for v in values]
                assert all(math.isfinite(v) for v in x), 'Non-finite loss at step '+str(step)
                losses[step-1]=x
                log.write(__import__('json').dumps(dict(step=step,V=x[0],Q=x[1],actor=x[2]),allow_nan=False)+'\n')
                if step%1000==0:
                    assert model.finite(), 'Non-finite model'
                    log.flush()
                    write(out/('progress_%06d.json'%step),dict(step=step,budget=job['steps'],pid=os.getpid(),
                        elapsed_seconds=time.monotonic()-started,finite=True,target_draws=sampler.draws,
                        target_stream_sha256=sampler.hash.hexdigest()))
                    print(job['id'],step,'/',job['steps'],'seconds',round(time.monotonic()-started,1),flush=True)
                if step==10000:
                    first10k=dict(step=step,parameter_sha256=model.digest(),
                        target_stream_sha256=sampler.hash.hexdigest(),target_draws=sampler.draws)
                    write(out/'FIRST_10000.json',first10k)
                if job['evaluate'] and step%10000==0:
                    before=rng_snapshot(torch,np,sampler)
                    result=science.evaluate(model,env,job)
                    after=rng_snapshot(torch,np,sampler)
                    assert before==after, 'Evaluation changed training or sampler RNG'
                    result['step']=step;result['length_statistics']=length_stats(result['episode_lengths'])
                    assert result['env_seeds']==list(range(500000,500010))
                    assert result['steps']==sum(result['episode_lengths'])
                    assert len(result['episode_lengths'])==10
                    evaluations.append(result);eval_rng.append(dict(step=step,before=before,after=after,identical=True))
                    write(out/('evaluation_%06d.json'%step),result)
        torch.cuda.synchronize()
        elapsed=time.monotonic()-started
        checkpoint=model.checkpoint()
        assert checkpoint['step']==job['steps']
        torch.save(checkpoint,str(out/'final_checkpoint.pt'))
        np.save(str(out/'losses.npy'),losses)
        write(out/'evaluations.json',dict(events=evaluations))
        write(out/'EVALUATION_RNG.json',dict(checks=eval_rng))
        write(out/'sampling.json',dict(target_first=sampler.first,target_draws=sampler.draws,
            target_stream_sha256=sampler.hash.hexdigest(),source_draws=0))
        completion=dict(completed=True,job=job,counts=counts,elapsed_seconds=elapsed,
            parameter_sha256=model.digest(),checkpoint_sha256=sha(out/'final_checkpoint.pt'),
            initialization_sha256=sha(out/'initialization.json'),losses_npy_sha256=sha(out/'losses.npy'),
            losses_jsonl_sha256=sha(out/'losses.jsonl'),sampling_sha256=sha(out/'sampling.json'),
            loss_statistics={k:loss_stats(losses[:,j]) for j,k in enumerate(('V','Q','actor'))},
            evaluation_count=len(evaluations),evaluation_episodes=sum(len(e['episode_lengths']) for e in evaluations),
            length_statistics=None if not evaluations else length_stats([n for e in evaluations for n in e['episode_lengths']]),
            all_model_parameters_finite=model.finite(),peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),ended_utc=utc(),
            formal_result=False,Last5_applicable=False,automatic_resume_supported=False)
        write(out/'COMPLETED.json',completion)
        print(job['id'],'completed',round(elapsed,2),'seconds',flush=True)
        return 0
    except BaseException:
        failure=traceback.format_exc()
        write(out/'FAILED.json',dict(time=utc(),traceback=failure,counts=counts,automatic_retry=False,
            unfinished_learner_step=inflight_step,
            failed_step_partial_Adam_updates_not_quantified=inflight_step is not None))
        print(failure,file=sys.stderr,flush=True)
        return 1
    finally:
        if env is not None:env.close()


if __name__=='__main__':raise SystemExit(main())
