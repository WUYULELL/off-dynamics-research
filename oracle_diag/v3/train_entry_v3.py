"""Approved GPU runs only. All changes to original train.py happen in memory."""
from support_v3 import *
from task_env import call_shifted_env,call_target_dataset,patch_normalization,paths
from methods_v3 import METHODS,IndexedReplayBuffer,build
import argparse,time,runpy,importlib,random,inspect,ctypes
import numpy as np,torch

def array_hash(a):
    a=np.ascontiguousarray(a);h=hashlib.sha256()
    h.update(str(a.dtype).encode());h.update(str(a.shape).encode());h.update(a.tobytes());return h.hexdigest()

def check_approval():
    file=OUT/'gpu_authorization.json'
    if not file.exists():raise RuntimeError('Stage B requires explicit user confirmation; GPU authorization has not been recorded.')
    auth=read(file)
    assert auth['authorized'] is True and auth['protocol_sha256']==sha(OUT/'protocol.json')
    gate=read(OUT/'stage_a_completed.json');assert gate['ready_for_training_review']
    for path,expected in gate['code_sha256'].items():assert sha(path)==expected,'V3 code changed since review: '+path
    return gate

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=METHODS,required=True)
    p.add_argument('--seed',type=int,choices=range(5),required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,choices=[5000,500000],default=500000)
    args=p.parse_args();gate=check_approval();out=args.out.resolve()
    if OUT not in out.parents:raise ValueError('Training output must be under the v3 result directory')
    fresh(out);torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required; CPU diagnostic fallback is not a training run')
    for key in list(os.environ):
        if key.startswith('ODRL_'):del os.environ[key]
    os.environ.update(ODRL_TARGET_FRACTION='1',ODRL_SOURCE_BUDGET='0.20',ODRL_BELLMAN_ETA='1.0')
    if args.method=='TARGET':os.environ['ODRL_TARGET_ONLY']='1'
    assert 'ODRL_PAPER_STYLE_SOURCE_TARGET' not in os.environ
    import algo.utils as utils
    utils.ReplayBuffer=IndexedReplayBuffer
    importlib.import_module('dataset.call_dataset').call_tar_dataset=call_target_dataset
    shifted_instances=[]
    def shifted(config):
        env=call_shifted_env(config);shifted_instances.append(env);return env
    importlib.import_module('envs.mujoco.call_mujoco_env').call_mujoco_env=shifted
    constants=patch_normalization('walker2d-morph');info=importlib.import_module('envs.infos');normal=info.get_normalized_score
    evaluations=[]
    def normalization(score,name):
        frame=inspect.currentframe().f_back
        assert frame.f_globals.get('__file__')==str(ROOT/'train.py')
        assert frame.f_locals['tar_eval_env'] is shifted_instances[0]
        assert score==frame.f_locals['tar_eval_return']
        value=normal(score,name);expected=(score-constants['min'])/(constants['max']-constants['min'])*100
        assert abs(value-expected)<1e-10
        evaluations.append(dict(step=int(frame.f_locals['t'])+1,raw_return=float(score),normalized_score=float(value),
            env_class=type(shifted_instances[0].unwrapped).__module__,xml_sha256=sha(paths('walker2d-morph')[0]),
            confirmed_target_env_object=True,lookup_key=name))
        write(out/'evaluation_verified.json',dict(normalization=constants,events=evaluations),replace=True)
        return value
    info.get_normalized_score=normalization
    oracle_receipt=Path(gate['oracle_receipt']);oracle=read(oracle_receipt)
    def factory(policy,config,mode,device):
        assert policy=='IQL' and mode==3 and str(device).startswith('cuda')
        assert config['batch_size']==128 and config['source_budget']==.20 and config['bellman_eta']==1.
        if args.steps==5000:config['eval_freq']=5000
        model=build(args.method,config,device,oracle_receipt);start=time.time();checked=False
        write(out/'effective_config.json',dict(config=config,seed=args.seed,method=args.method,oracle_receipt=str(oracle_receipt),
            oracle_receipt_sha256=sha(oracle_receipt),protocol_sha256=sha(OUT/'protocol.json'),
            standard_direction=True,total_batch=128,source_draws=0 if args.method=='TARGET' else 64,
            target_draws=128 if args.method=='TARGET' else 64,normalization=constants,
            purpose='smoke; excluded from formal results' if args.steps==5000 else 'formal',
            tensorboard_target_score_tag='test/target normalized score'))
        digest=hashlib.sha256()
        for net in [model.policy,model.v_func,model.q_funcs]:
            for val in net.state_dict().values():digest.update(val.detach().cpu().numpy().tobytes())
        write(out/'initialization.json',dict(seed=args.seed,sha256=digest.hexdigest()))
        base=model.train
        def instrument(src,tar,batch_size=128,writer=None):
            nonlocal checked
            if not checked:
                assert src.size==oracle['n'] and tar.size==5000 and batch_size==128
                for attr,key in [('state','observations'),('action','actions'),('next_state','next_observations'),('reward','rewards'),('not_done','terminals')]:
                    val=getattr(src,attr)
                    if key in ['rewards','terminals']:val=val.ravel()
                    if key=='terminals':val=(1.-val).astype(bool)
                    assert array_hash(val)==oracle['source_array_sha256'][key],key
                expected=call_target_dataset('walker2d-morph')
                for attr,key in [('state','observations'),('action','actions'),('next_state','next_observations'),('reward','rewards'),('not_done','terminals')]:
                    val=getattr(tar,attr).reshape(expected[key].shape)
                    assert np.array_equal(val,1.-expected[key] if key=='terminals' else expected[key]),key
                write(out/'data_verified.json',dict(source_n=src.size,target_n=tar.size,all_source_rows_aligned=True,
                    all_target_rows_equal_file=True,source_dataset_sha256=oracle['source_dataset_sha256'],target_dataset_sha256=sha(paths('walker2d-morph')[1])))
                np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);random.seed(args.seed)
                checked=True
            before=(src.draw_count,tar.draw_count);base(src,tar,batch_size,writer);step=model.total_it
            actual=(src.draw_count-before[0],tar.draw_count-before[1]);wanted=(0,128) if args.method=='TARGET' else (64,64)
            assert actual==wanted
            if step==1:write(out/'first_sample.json',dict(source_ids=getattr(src,'last_indices',np.array([],int)).tolist(),
                target_ids=tar.last_indices.tolist(),actual_source_draws=actual[0],actual_target_draws=actual[1]))
            if step%5000==0 or step==args.steps:
                assert all(torch.isfinite(v).all() for net in [model.policy,model.v_func,model.q_funcs] for v in net.parameters())
                write(out/'progress.json',dict(step=step,elapsed=time.time()-start,pid=os.getpid(),
                    source_draws=src.draw_count,target_draws=tar.draw_count,gpu_peak_bytes=torch.cuda.max_memory_allocated()),replace=True)
            if step%100000==0 or step==args.steps:
                (out/'values').mkdir(exist_ok=True);torch.save(model.v_func.state_dict(),winpath(out/'values'/('V_%d.pt'%step)))
        model.train=instrument;return model
    importlib.import_module('algo.call_algo').call_algo=factory
    short=ctypes.create_unicode_buffer(32768)
    if not ctypes.windll.kernel32.GetShortPathNameW(str(out),short,len(short)):raise RuntimeError('Cannot create short Windows output path')
    params=dict(batch_size=128,source_budget=.20,source_loss_weight=1.,bellman_eta=1.,target_only=args.method=='TARGET')
    sys.argv=['train.py','--policy','IQL','--env','walker2d-morph','--shift_level','medium','--srctype','medium','--tartype','medium',
        '--mode','3','--seed',str(args.seed),'--max_step',str(args.steps),'--dir',short.value,'--save-model','--params',json.dumps(params)]
    runpy.run_path(str(ROOT/'train.py'),run_name='__main__')
    assert evaluations and evaluations[-1]['step']==args.steps
    write(out/'completed.json',dict(steps=args.steps,method=args.method,seed=args.seed,task='walker2d-morph',ended=time.time()))

if __name__=='__main__':main()
