"""Whitelisted Stage0 diagnostics and Stage1 formal jobs; single-GPU controller only."""
from common import *
import argparse
import traceback
import numpy as np
import torch
from actor import initialize, Sampler
from learner import initialize as original_initialize
from train import evaluate

def main(trial_id, formal=False):
    guard(); assert r.read(OUT/'CPU_ACCEPTANCE.json')['passed']; r.configure_cpu()
    assert not competing_processes(), 'Another experimental queue/train is active'
    assert torch.cuda.is_available(), 'No CPU fallback'
    if formal:
        assert r.read(OUT/'STAGE0_COMPLETED.json')['passed']
        assert r.read(OUT/'DELIVERY_VERIFIED.json')['passed']
        job=next(j for j in formal_jobs() if j['id']==trial_id)
        variant,wa,steps,run_seed=job['variant'],1.,500000,job['seed']
        out=Path(job['out'])
    else:
        _,variant,wa,steps=next(x for x in TRIALS if x[0]==trial_id)
        run_seed=201;out=OUT/'runs'/trial_id
    r.fresh(out)
    c = dict(settings()['learner'],seed=run_seed)
    r.write(out/'config.json', dict(learner=c, variant=variant, actor_source_w=wa,
        steps_executed=steps, cosine_T_max=500000, seed=run_seed, scope='Stage1_Hopper_formal' if formal else 'Stage0_diagnostic_only',
        scientific_config_sha256=r.sha(OUT/'config.json'), correction=settings()['correction'],
        source_top_reduction=settings()['source_top_reduction']))
    r.write(out/'started.json', dict(pid=os.getpid(),time=time.time()))
    try:
        arrays_t = r.target_data(TASK); arrays_s, ids = r.source_data(TASK)
        model = initialize(c,'cuda:0',variant,wa)
        original = original_initialize(c,'cuda:0') if wa == 0. else None
        fixture = r.read(r.OUT/'PAIRING_EXPECTATIONS.json')['seeds'][str(run_seed)]['initialization']
        assert model.digest() == fixture
        if original is not None: assert original.digest() == fixture
        r.write(out/'initialization.json',dict(sha256=fixture,original_O1_match=True))
        target=Sampler(arrays_t,run_seed,'target_sample','cuda:0',task=TASK)
        source=Sampler(arrays_s,run_seed,'source_sample','cuda:0',task=TASK)
        env=r.env(TASK); events=[]; eval_seconds=0.; evaluation_steps=0
        historical = r.read(r.OUT/'runs/hopp_O1_s201/early_losses.json')['trace']
        vectors = np.lib.format.open_memmap(r.winpath(out/'diagnostic_vectors.npy'),mode='w+',dtype=np.float32,
                    shape=(steps,64,6)) if wa else None
        writer=None
        if formal:
            from tensorboardX import SummaryWriter
            writer=SummaryWriter(str(Path(r.shortpath(out))/'tb'))
        torch.cuda.synchronize(); start=time.perf_counter()
        with open(r.winpath(out/'steps.jsonl'),'x',encoding='utf-8') as log:
            for step in range(1,steps+1):
                tb,sb=target.sample(),source.sample()
                baseline=original.step(tb,sb) if original is not None else None
                losses=model.step(tb,sb)
                values=[float(x) for x in losses]
                record=dict(step=step,V=values[0],Q=values[1],actor=values[2])
                if baseline is not None:
                    baseline_values=[float(x) for x in baseline]
                    assert all(torch.equal(a,b) for a,b in zip(losses,baseline)), (step,values,baseline_values)
                    record.update(O1_V=baseline_values[0],O1_Q=baseline_values[1],O1_actor=baseline_values[2],
                                  losses_bitwise_equal=True)
                    if step<=10:
                        old=historical[step-1]
                        assert all(np.isclose(record[k],old[k],rtol=1e-6,atol=1e-7) for k in ('V','Q','actor'))
                else:
                    record.update(model.diagnostics)
                    vectors[step-1]=model.diagnostic_vectors
                assert all(np.isfinite(v) for v in record.values() if isinstance(v,(float,int)))
                log.write(json.dumps(record,allow_nan=False,separators=(',',':'))+'\n')
                if step%1000==0:
                    assert full_finite(scientific_state(model))
                    log.flush()
                    r.write(out/'progress.json',dict(step=step,budget=steps,cosine_T_max=500000,pid=os.getpid(),
                        elapsed=time.perf_counter()-start,finite=True,target_draws=target.draws,source_draws=source.draws,
                        target_stream_sha256=target.hash.hexdigest(),source_stream_sha256=source.hash.hexdigest()),replace=True)
                if step%10000==0:
                    torch.cuda.synchronize(); t=time.perf_counter()
                    result=evaluate(model,env,dict(seed=run_seed,task=TASK)); result['step']=step
                    torch.cuda.synchronize(); duration=time.perf_counter()-t
                    result['elapsed_seconds']=duration;eval_seconds+=duration;evaluation_steps+=result['steps']
                    events.append(result);r.write(out/'evaluations.json',dict(events=events),replace=True)
                    if writer:
                        writer.add_scalar('evaluation/target_score',result['mean'],step)
                        for key in ('source_A_positive_fraction','target_weight_sum','source_weight_sum','source_selected'):
                            writer.add_scalar('diagnostics/'+key,model.diagnostics[key],step)
                        writer.flush()
                if formal and step%100000==0:
                    torch.save(model.checkpoint(),r.winpath(out/('model_%d.pt'%step)))
            log.flush();os.fsync(log.fileno())
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        env.close()
        if writer:writer.close()
        if vectors is not None: vectors.flush();del vectors
        model_path=out/'model_final.pt';torch.save(model.checkpoint(),r.winpath(model_path))
        state=scientific_state(model);assert full_finite(state)
        target_expected=r.stream_hash(run_seed,'target_sample',5000,steps,TASK)
        source_expected=r.stream_hash(run_seed,'source_sample',999998,steps,TASK)
        assert target.hash.hexdigest()==target_expected and source.hash.hexdigest()==source_expected
        sampling=dict(target_draws=target.draws,source_draws=source.draws,target_first=target.first,source_first=source.first,
            target_stream_sha256=target_expected,source_stream_sha256=source_expected,
            canonical_ids_sha256=hashlib.sha256(ids.astype('<i8').tobytes()).hexdigest())
        if formal:
            expected=r.read(r.OUT/'PAIRING_EXPECTATIONS.json')['seeds'][str(run_seed)]
            assert target_expected==expected['target'] and source_expected==expected['source']
        r.write(out/'sampling.json',sampling)
        if original is not None:
            assert tree_digest(state)==tree_digest(scientific_state(original))
            assert model.digest()==original.digest()
            original_path=out/'original_O1_model_final.pt'
            torch.save(original.checkpoint(),r.winpath(original_path))
            r.write(out/'PARITY_RESULT.json',dict(passed=True,steps=steps,
                every_step_losses_bitwise_equal=True,final_4network_parameters_bitwise_equal=True,
                final_3Adam_and_cosine_bitwise_equal=True,model_digest=model.digest(),
                scientific_state_sha256=tree_digest(state),original_scientific_state_sha256=tree_digest(scientific_state(original)),
                original_model=str(original_path),original_model_sha256=r.sha(original_path),
                historical_first10_loss_match=True,reference_learner_updates=10000))
        inputs={str(p):r.sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='progress.json'}
        r.write(out/'completed.json',dict(passed=True,trial_id=trial_id,steps=steps,finite=True,
            model_digest=model.digest(),scientific_state_sha256=tree_digest(state),critic_state_sha256=tree_digest(critic_state(model)),
            elapsed_seconds=elapsed,training_and_logging_seconds=elapsed-eval_seconds,evaluation_seconds=eval_seconds,
            evaluation_episodes=len(events)*10,evaluation_steps=evaluation_steps,
            new_learner_updates=steps,reference_learner_updates=steps if original is not None else 0,
            Adam_steps=3*steps*(2 if original is not None else 1),sha256=inputs,time=time.time()))
    except BaseException:
        r.write(out/'failed.json',dict(time=time.time(),traceback=traceback.format_exc(),no_automatic_retry=True))
        raise

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--trial',required=True);parser.add_argument('--formal',action='store_true')
    args=parser.parse_args();main(args.trial,args.formal)
