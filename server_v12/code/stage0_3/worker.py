"""One fixed benchmark job with common train/evaluation barriers; never a formal run."""
import os
import sys
sys.dont_write_bytecode=True
os.environ['PYTHONDONTWRITEBYTECODE']='1'
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'
os.environ.pop('ODRL_PAPER_STYLE_SOURCE_TARGET',None)
import argparse
import hashlib
import json
import math
from pathlib import Path
import pickle
import random
import time
import traceback
from common import HERE,read,write,publish,sha,utc,verify_package,private_binding,length_stats,gpu_info,loss_stats
from science import Science


def wait_marker(path,timeout,parent_pid):
    end=time.monotonic()+timeout
    while not path.exists():
        assert os.getppid()==parent_pid,'Controller exited; stop instead of retaining an orphan GPU worker'
        if time.monotonic()>end:raise TimeoutError('Barrier timed out: '+str(path))
        time.sleep(.05)
    return read(path)


def rng_state(torch,np,sampler):
    return dict(Python=hashlib.sha256(pickle.dumps(random.getstate(),protocol=4)).hexdigest(),
        NumPy=hashlib.sha256(pickle.dumps(np.random.get_state(),protocol=4)).hexdigest(),
        CPU=sha_bytes(torch.get_rng_state().numpy().tobytes()),
        CUDA=[sha_bytes(v.cpu().numpy().tobytes()) for v in torch.cuda.get_rng_state_all()],
        sampler=sha_bytes(pickle.dumps(sampler.rng.get_state(),protocol=4)),draws=sampler.draws,stream=sampler.hash.hexdigest())


def sha_bytes(value):return hashlib.sha256(value).hexdigest()


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--job',type=Path,required=True);args=ap.parse_args()
    verify_package();job_path=args.job.resolve();job=read(job_path);base=job_path.parent.parent
    auth=read(base/'WORKER_AUTHORIZATION.json')
    assert auth['scope']=='V12_0.3_benchmark_only' and auth['jobs'][job['id']]==sha(job_path)
    assert job['algorithm'] in ('IQL','BC') and job['task']=='hopper-morph'
    assert 901<=job['seed']<=906 and job['steps']==20000
    assert job['algorithm']=='IQL' or (job['seed']==901 and job['concurrency']==1 and job['variant']=='BC_all')
    assert job['gpu_uuid']==os.environ['CUDA_VISIBLE_DEVICES'] and job['gpu_uuid'].startswith('GPU-')
    assert ',' not in os.environ['CUDA_VISIBLE_DEVICES'] and sys.version_info[:3]==(3,8,13) and sys.platform=='linux'
    out=base/'runs'/job['id'];out.mkdir()
    tier=base/'tiers'/job['tier'];counts=dict(training_updates=0,Adam_updates=0,constructor_environment_steps=0,evaluation_environment_steps=0)
    env=None;inflight=None
    try:
        for name in ('tmp','cache','cuda_cache'):(out/name).mkdir()
        for key,value in {'TMPDIR':out/'tmp','XDG_CACHE_HOME':out/'cache','CYTHON_CACHE_DIR':out/'cache/cython','CUDA_CACHE_PATH':out/'cuda_cache'}.items():os.environ[key]=str(value)
        import numpy as np
        import torch
        torch.set_num_threads(1);torch.set_num_interop_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=True;torch.backends.cudnn.allow_tf32=True
        s=Science(Path(job['project']));s.load()
        data=s.r.target_data('hopper-morph')
        assert torch.__version__=='1.11.0+cu113' and torch.cuda.is_available() and torch.cuda.device_count()==1
        assert 'A6000' in torch.cuda.get_device_name(0)
        if job['algorithm']=='IQL':
            config=s.iql_config(job['seed']);model=s.iql['initialize'](config,'cuda:0')
            sampler=s.iql['Sampler'](data,job['seed'],'target_sample','cuda:0',task='hopper-morph')
            evaluate=s.iql_evaluate
        else:
            config=dict(actor_lr=3e-4,hidden_sizes=256,batch=64,beta=0,cosine_max_step=200000,Q_V_constructed=False)
            assert s.bc_runtime.ids(job)==list(range(5000))
            model=s.bc['Actor'](job,'cuda:0');sampler=s.bc['Sampler'](data,job,'cuda:0');evaluate=s.bc_evaluate
        write(out/'config.json',dict(job=job,learner=config,physical_gpu=gpu_info(job['gpu_uuid']),
            scientific_AST=s.science,threads=1,source_rows=0,raw_observations=True,reward_scale=1.,
            diagnostic_only=True,no_warmup_optimization=True,deterministic_algorithms=True,
            matmul_allow_tf32=True,cudnn_allow_tf32=True,CUBLAS_WORKSPACE_CONFIG=':4096:8'))
        write(out/'initialization.json',dict(parameter_sha256=model.digest()))
        private=private_binding(out);os.environ['MUJOCO_PY_FORCE_CPU']='1'
        import mujoco_py
        assert private in Path(mujoco_py.__file__).resolve().parents
        extension=Path(mujoco_py.cymj.__file__).resolve();assert private in extension.parents
        write(out/'NATIVE_RUNTIME.json',dict(extension=str(extension),extension_sha256=sha(extension)))
        import gym
        from gym.envs.mujoco.hopper_v3 import HopperEnv
        raw=HopperEnv(xml_file=s.task_spec('hopper-morph')['xml']);counts['constructor_environment_steps']=1
        original_step=raw.step
        def counted(action):
            counts['evaluation_environment_steps']+=1
            return original_step(action)
        raw.step=counted;env=gym.wrappers.TimeLimit(raw,max_episode_steps=1000)
        torch.cuda.synchronize()
        publish(out/'READY.json',dict(pid=os.getpid(),time=utc(),gpu_uuid=job['gpu_uuid']))
        release=wait_marker(tier/'TRAIN_START.json',job['barrier_timeout_seconds'],job['controller_pid'])
        torch.cuda.synchronize();begin=time.monotonic();mid=None
        losses=np.empty((20000,3 if job['algorithm']=='IQL' else 1),dtype=np.float32)
        with (out/'losses.jsonl').open('x',encoding='utf-8') as log:
            for step in range(1,20001):
                inflight=step
                if job['algorithm']=='IQL':values=model.step(sampler.sample(),None);adam=3
                else:values=(model.step(*sampler.sample()),);adam=1
                counts['training_updates']+=1;counts['Adam_updates']+=adam;inflight=None
                vals=[float(v) for v in values];assert all(math.isfinite(v) for v in vals)
                losses[step-1]=vals
                log.write(json.dumps(dict(step=step,losses=vals),allow_nan=False)+'\n')
                if step%1000==0:
                    assert os.getppid()==job['controller_pid'],'Controller exited; stop this worker'
                    assert model.finite();log.flush()
                    write(out/('progress_%06d.json'%step),dict(step=step,elapsed=time.monotonic()-begin,pid=os.getpid(),finite=True))
                if step==10000:torch.cuda.synchronize();mid=time.monotonic()
                if step in (10000,20000):print(job['id'],step,'/ 20000',flush=True)
        torch.cuda.synchronize();end=time.monotonic()
        trained=dict(training_updates=20000,train_start_monotonic=begin,train_end_monotonic=end,
            release_monotonic=release['monotonic'],training_seconds=end-begin,
            first_10000_seconds=mid-begin,second_10000_seconds=end-mid,
            target_draws=sampler.draws,target_stream_sha256=sampler.hash.hexdigest())
        publish(out/'TRAINING_DONE.json',trained)
        wait_marker(tier/'EVALUATION_START.json',job['barrier_timeout_seconds'],job['controller_pid'])
        before=rng_state(torch,np,sampler);t=time.monotonic();evaluation=evaluate(model,env,job)
        eval_seconds=time.monotonic()-t;after=rng_state(torch,np,sampler);assert before==after
        evaluation.update(step=20000,length_statistics=length_stats(evaluation['episode_lengths']),evaluation_seconds=eval_seconds)
        write(out/'evaluation.json',evaluation);write(out/'EVALUATION_RNG.json',dict(before=before,after=after,identical=True))
        torch.save(model.checkpoint(),str(out/'final_checkpoint.pt'));np.save(str(out/'losses.npy'),losses)
        write(out/'sampling.json',dict(target_draws=sampler.draws,target_stream_sha256=sampler.hash.hexdigest(),
            source_draws=0,target_first=getattr(sampler,'first',None)))
        completion=dict(trained,completed=True,job=job,counts=counts,evaluation_seconds=eval_seconds,
            evaluation_episodes=10,length_statistics=evaluation['length_statistics'],
            parameter_sha256=model.digest(),all_model_parameters_finite=model.finite(),
            peak_GPU_allocated_bytes=torch.cuda.max_memory_allocated(),peak_GPU_reserved_bytes=torch.cuda.max_memory_reserved(),
            losses_sha256=sha(out/'losses.npy'),JSONL_sha256=sha(out/'losses.jsonl'),checkpoint_sha256=sha(out/'final_checkpoint.pt'),
            loss_statistics=[loss_stats(losses[:,j]) for j in range(losses.shape[1])],ended_utc=utc(),
            formal_result=False,Last5_applicable=False)
        write(out/'COMPLETED.json',completion);return 0
    except BaseException:
        error=traceback.format_exc();write(out/'FAILED.json',dict(time=utc(),traceback=error,counts=counts,
            unfinished_update=inflight,partial_failed_Adam_updates_not_quantified=inflight is not None,automatic_retry=False))
        print(error,file=sys.stderr,flush=True);return 1
    finally:
        if env is not None:env.close()


if __name__=='__main__':raise SystemExit(main())
