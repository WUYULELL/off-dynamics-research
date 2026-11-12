"""V13 BC training and CPU audit of complete BC attempts."""
import argparse
import hashlib
import math
import os
from pathlib import Path
import resource
import struct
import time
import traceback

import numpy as np

from common import length_stats, read, sha, utc, write
from science import Project, bc_definitions, rows_digest, transitions, validate_dimensions
from stage1_core import CONFIG, PROJECT, TASK, cache_receipt, publish, verify_cache_files, verify_worker
from v12_bc import EVAL_POINTS, STEPS, configure_torch, eval_records, finite_state, rng_digest


def inputs(job):
    project = Project(PROJECT)
    spec = project.spec(TASK)
    cache = Path(job['cache'])
    assert sha(cache/'CACHE_ACCEPTANCE.json') == job['cache_receipt_sha256']
    receipt = cache_receipt(cache)
    variant = job['variant']
    payload, indices, digests = {}, {}, {}
    if variant == 'BC-all' or variant.startswith('BC-mix'):
        target_path = project.path(spec['target_dataset'])
        arrays, rows = transitions(target_path, actor_only=True)
        validate_dimensions(arrays, spec)
        assert len(rows) == CONFIG['target_rows']
        payload['target'] = arrays['observations'],arrays['actions']
        indices['target'] = rows
        digests['target'] = dict(rows=len(rows),sample_index_sha256=rows_digest(rows),data_sha256=sha(target_path))
    if variant != 'BC-all':
        names = ['source_observations.npy','action_fixed.npy','row_id.npy']
        filtered = variant.endswith('-sel')
        if filtered:
            names.append('selected_positions.npy')
        verify_cache_files(cache,receipt,names)
        n = CONFIG['canonical_source_rows']
        state,action,raw_ids = [np.load(cache/name,allow_pickle=False,mmap_mode='r') for name in names[:3]]
        assert state.shape == (n,11) and action.shape == (n,3) and raw_ids.shape == (n,)
        if filtered:
            rows = np.load(cache/names[-1],allow_pickle=False,mmap_mode='r')
            assert len(rows) == receipt['selected_rows'] and np.all(rows[1:]>rows[:-1])
            assert rows[0]>=0 and rows[-1]<n
        else:
            rows = np.arange(n,dtype=np.int64)
        payload['source'] = state,action
        indices['source'] = rows
        digests['source'] = dict(rows=len(rows),sample_index_sha256=rows_digest(rows),raw_row_sha256=rows_digest(raw_ids[rows]),
            files={name:receipt['cache_sha256'][name] for name in names})
    return project,payload,indices,digests


def wait_for_release(job, out):
    ready = out/'RELEASE.json'
    while not ready.is_file():
        assert os.getppid() == job['controller_pid'], 'Stage 1 controller exited before GPU release'
        time.sleep(.2)
    assert read(ready)['gpu_uuid'] == job['gpu']['uuid']
    assert os.getppid() == job['controller_pid']


def one_update(actor, samplers, torch):
    if len(samplers) == 1:
        domain,sampler = next(iter(samplers.items()))
        value = float(actor.step(*sampler.sample()))
        return (value,0.,value) if domain == 'target' else (0.,value,value)
    actor.total_it += 1
    state,action = samplers['target'].sample()
    target = actor.policy.bc_loss(state,action).mean(dim=1,keepdim=True).mean()
    state,action = samplers['source'].sample()
    source = actor.policy.bc_loss(state,action).mean(dim=1,keepdim=True).mean()
    loss = target+source
    actor.optimizer.zero_grad()
    loss.backward()
    actor.optimizer.step()
    actor.scheduler.step()
    return tuple(torch.stack((target.detach(),source.detach(),loss.detach())).cpu().tolist())


def train(job, out):
    expected = CONFIG['bc']
    assert job['seed'] in expected['seeds'] and job['variant'] in expected['variants']
    assert os.environ['CUDA_VISIBLE_DEVICES'] == job['gpu']['uuid'] and ',' not in job['gpu']['uuid']
    wait_for_release(job,out)
    import mujoco_py
    import torch
    from tensorboardX import SummaryWriter

    binding = Path(job['private_binding']).resolve()
    assert binding in Path(mujoco_py.cymj.__file__).resolve().parents
    assert torch.__version__ == '1.11.0+cu113'
    configure_torch(torch)
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    assert 'A6000' in torch.cuda.get_device_name(0)
    project,payload,indices,inputs_proof = inputs(job)
    functions = bc_definitions(project,np.arange(CONFIG['target_rows'],dtype=np.int64))
    functions['ids'] = lambda domain_job: indices[domain_job['domain']]
    actor = functions['Actor'](job,'cuda:0')
    samplers = {}
    for domain,data in payload.items():
        sampler = functions['Sampler'](data,dict(job,domain=domain),'cuda:0')
        if domain == 'source':
            sampler.rng = np.random.RandomState(functions['seed']('source_sample',job['seed'],TASK))
        samplers[domain] = sampler
    write(out/'config.json',dict(job=job,training=expected,filter=CONFIG['residual_filter'],inputs=inputs_proof,
                                 normalization=project.spec(TASK),Q_or_V_constructed=False))
    write(out/'INITIALIZATION.json',dict(parameter_sha256=actor.digest()))
    env = functions['env'](TASK)
    assert env.unwrapped.frame_skip == 4 and env._max_episode_steps == 1000
    writer = SummaryWriter(str(out/'tb'))
    events = []
    start = time.monotonic()
    training_seconds = 0.
    block_start = start
    try:
        with (out/'steps.f32').open('xb') as step_log:
            for step in range(1,STEPS+1):
                values = one_update(actor,samplers,torch)
                if not all(math.isfinite(x) for x in values):
                    raise FloatingPointError(f'Nonfinite BC loss at {step}')
                step_log.write(struct.pack('<fff',*values))
                if step%1000 == 0:
                    assert os.getppid() == job['controller_pid'], 'Stage 1 controller exited during training'
                    if not actor.finite():
                        raise FloatingPointError(f'Nonfinite BC actor at {step}')
                    step_log.flush()
                    os.fsync(step_log.fileno())
                    publish(out/'heartbeat.json',dict(step=step,total=STEPS,pid=os.getpid(),time=utc(),
                        elapsed_seconds=time.monotonic()-start,loss=values[2]))
                    writer.add_scalar('loss/actor',values[2],step)
                if step in EVAL_POINTS:
                    torch.cuda.synchronize()
                    training_seconds += time.monotonic()-block_start
                    t = time.monotonic()
                    before = rng_digest(torch)
                    event = functions['evaluate'](actor,env,job)
                    after = rng_digest(torch)
                    assert before == after
                    event.update(step=step,training_RNG_sha256_before=before,training_RNG_sha256_after=after,
                                 length_statistics=length_stats(event['episode_lengths']),evaluation_seconds=time.monotonic()-t)
                    write(out/f'eval_{step:06d}.json',event)
                    events.append(event)
                    writer.add_scalar('evaluation/target_score',event['mean'],step)
                    writer.flush()
                    print(job['job'],step,'/',STEPS,flush=True)
                    block_start = time.monotonic()
    finally:
        writer.close()
        env.close()
    checkpoint = out/'final_checkpoint.pt'
    torch.save(actor.checkpoint(),str(checkpoint))
    write(out/'COMPLETED.json',dict(job=job,steps=actor.total_it,Adam_updates=actor.total_it,finite=actor.finite(),
        training_seconds=training_seconds,wall_seconds=time.monotonic()-start,
        peak_RSS_bytes=1024*resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        evaluation_episodes=sum(len(event['episode_lengths']) for event in events),
        evaluation_environment_steps=sum(event['steps'] for event in events),
        evaluation_seconds=sum(event['evaluation_seconds'] for event in events),
        checkpoint_sha256=sha(checkpoint),model_digest=actor.digest(),steps_sha256=sha(out/'steps.f32'),
        eval_sha256={f'eval_{p:06d}.json':sha(out/f'eval_{p:06d}.json') for p in EVAL_POINTS},
        streams={d:dict(draws=s.draws,sha256=s.hash.hexdigest()) for d,s in samplers.items()},
        GPU_peak_allocated_bytes=torch.cuda.max_memory_allocated(),ended_utc=utc()))


def audit(job, out):
    import torch
    assert os.environ['CUDA_VISIBLE_DEVICES'] == '-1'
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    done = read(out/'COMPLETED.json')
    cfg = read(out/'config.json')
    assert done['job'] == cfg['job'] == job and done['steps'] == done['Adam_updates'] == STEPS
    assert done['finite'] and cfg['training'] == CONFIG['bc'] and cfg['filter'] == CONFIG['residual_filter']
    assert cfg['Q_or_V_constructed'] is False
    checkpoint = out/'final_checkpoint.pt'
    assert sha(checkpoint) == done['checkpoint_sha256']
    ck = torch.load(str(checkpoint),map_location='cpu')
    assert set(ck) == {'policy','actor_optimizer','actor_scheduler','step','job'}
    assert ck['job'] == job and ck['step'] == STEPS and finite_state(torch,ck)
    assert ck['actor_scheduler']['T_max'] == ck['actor_scheduler']['last_epoch'] == STEPS
    adam = ck['actor_optimizer']
    assert len(adam['state']) == len(ck['policy']) and all(int(x['step']) == STEPS for x in adam['state'].values())
    assert adam['param_groups'][0]['lr'] == 0.
    digest = hashlib.sha256(b''.join(x.numpy().tobytes() for x in ck['policy'].values())).hexdigest()
    assert digest == done['model_digest']
    project, _, indices, input_proof = inputs(job)
    assert input_proof == cfg['inputs'] and set(indices) == set(done['streams'])
    functions = bc_definitions(project,np.arange(CONFIG['target_rows'],dtype=np.int64))
    initial = functions['Actor'](job,'cpu').digest()
    assert initial == read(out/'INITIALIZATION.json')['parameter_sha256']
    for domain,ids in indices.items():
        rng = np.random.RandomState(functions['seed'](domain+'_sample',job['seed'],TASK))
        stream = hashlib.sha256()
        for offset in range(0,STEPS,4096):
            count = min(4096,STEPS-offset)*64
            stream.update(ids[rng.randint(len(ids),size=count)].astype('<i8').tobytes())
        assert done['streams'][domain] == dict(draws=STEPS*64,sha256=stream.hexdigest())
    loss_path = out/'steps.f32'
    assert sha(loss_path) == done['steps_sha256']
    loss = np.fromfile(loss_path,dtype='<f4').reshape(-1,3)
    assert loss.shape == (STEPS,3) and np.isfinite(loss).all() and (loss>=0).all()
    assert np.array_equal(loss[:,0]+loss[:,1],loss[:,2])
    for column,domain in enumerate(('target','source')):
        if domain not in indices:
            assert (loss[:,column] == 0).all()
    for name,digest in done['eval_sha256'].items():
        assert sha(out/name) == digest
    assert set(done['eval_sha256']) == {f'eval_{p:06d}.json' for p in EVAL_POINTS}
    records = eval_records(out,job,project.spec(TASK))
    assert done['evaluation_episodes'] == 400 and done['evaluation_environment_steps'] == sum(e['steps'] for e in records)
    assert not torch.cuda.is_initialized()
    write(out/'AUDIT.json',dict(passed=True,CPU_only=True,complete_sample_streams_verified=True,
        full_loss_log_verified=True,raw_evaluation_points=len(records),checkpoint_sha256=sha(checkpoint),
        completed_sha256=sha(out/'COMPLETED.json'),time=utc()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('train','audit'))
    parser.add_argument('job',type=Path)
    args = parser.parse_args()
    job = verify_worker(args.job,1)
    folder = Path(job['output'])
    try:
        (train if args.mode == 'train' else audit)(job,folder)
    except Exception as exc:
        name = 'FAILED.json' if args.mode == 'train' else 'AUDIT_FAILED.json'
        write(folder/name,dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc(),time=utc()))
        raise
