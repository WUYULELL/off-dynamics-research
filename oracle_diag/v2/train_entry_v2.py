"""Run original train.py with isolated source loader/algorithm instrumentation."""
from support import *
import argparse,time,runpy,importlib,random
from fixed_loader import load_source_fixed

def main():
    p=argparse.ArgumentParser();p.add_argument('--method',choices=['TARGET','CONST','BACKUP','MODEL'],required=True)
    p.add_argument('--seed',type=int,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--steps',type=int,default=500000);p.add_argument('--check-eval',type=int,default=0)
    args=p.parse_args();out=args.out.resolve();fresh(out)
    import numpy as np,torch
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required')
    import algo.utils as utils
    from oracle_methods import IndexedReplayBuffer,IQL_ORACLE_BACKUP
    utils.ReplayBuffer=IndexedReplayBuffer
    dataset=importlib.import_module('dataset.call_dataset')
    def fixed(name,level,quality):
        assert name=='walker2d-kinematic-footjnt','Unexpected source load'
        return load_source_fixed(name,level,quality)
    dataset.call_tar_dataset=fixed
    env=ENVS[0];oracle_path=OUT/'stage_a'/(env+'_source_oracle_fixed.npz')
    if args.method=='MODEL':
        b=read(OUT/'stage_b/completed.json');assert b['selected'];oracle_path=Path(b['selected_file'])
    oracle=dict(np.load(oracle_path));registry=importlib.import_module('algo.call_algo');original=registry.call_algo
    def factory(policy,config,mode,device):
        if args.check_eval:config['eval_freq']=args.check_eval
        model=IQL_ORACLE_BACKUP(config,device,oracle) if args.method in ['BACKUP','MODEL'] else original(policy,config,mode,device)
        config['seed']=args.seed
        write(out/'effective_config.json',dict(config=config,seed=args.seed,method=args.method,
            oracle_path=str(oracle_path),oracle_sha256=sha(oracle_path),sampler='original np.random randint; shared seed reset at first training update'))
        digest=hashlib.sha256()
        for net in [model.policy,model.v_func,model.q_funcs]:
            for val in net.state_dict().values():digest.update(val.detach().cpu().numpy().tobytes())
        write(out/'initialization.json',dict(seed=args.seed,sha256=digest.hexdigest()))
        base=model.train;checked=False;start=time.time()
        def instrument(src,tar,batch_size=128,writer=None):
            nonlocal checked
            if not checked:
                assert src.size==5000 and batch_size==128
                for field,key in [('state','source_state'),('action','source_action'),('next_state','source_next'),('reward','source_reward'),('not_done','source_not_done')]:
                    assert np.array_equal(getattr(src,field).reshape(oracle[key].shape),oracle[key]),field
                assert config.get('source_budget',.2)==.2 and config.get('bellman_eta',1.)==1.
                th=hashlib.sha256()
                for field in ['state','action','next_state','reward','not_done']:th.update(getattr(tar,field).tobytes())
                write(out/'data_verified.json',dict(source_n=src.size,target_n=tar.size,aligned=True,target_sha256=th.hexdigest()))
                # Common training start, independent of different initial evaluation lengths.
                np.random.seed(args.seed);torch.manual_seed(args.seed);torch.cuda.manual_seed_all(args.seed);random.seed(args.seed)
                checked=True
            base(src,tar,batch_size,writer);step=model.total_it
            if step==1:
                write(out/'first_sample.json',dict(source_ids=src.last_indices.tolist() if hasattr(src,'last_indices') else None,target_ids=tar.last_indices.tolist()))
            if step%5000==0 or step==args.steps:
                assert all(torch.isfinite(v).all() for net in [model.policy,model.v_func,model.q_funcs] for v in net.parameters()),'Nonfinite parameters'
                write(out/'progress.json',dict(step=step,elapsed=time.time()-start,pid=os.getpid()),replace=True)
            if step%100000==0 or step==args.steps:
                (out/'values').mkdir(exist_ok=True);torch.save(model.v_func.state_dict(),str(out/'values'/('V_%d.pt'%step)))
        model.train=instrument;return model
    registry.call_algo=factory
    for key in list(os.environ):
        if key.startswith('ODRL_'):del os.environ[key]
    os.environ.update(ODRL_PAPER_STYLE_SOURCE_TARGET='1',ODRL_TARGET_FRACTION='0.1',ODRL_SOURCE_BUDGET='0.20',ODRL_BELLMAN_ETA='1.0',ODRL_TARGET_ONLY='1' if args.method=='TARGET' else '0')
    policy={'TARGET':'IQL','CONST':'IQL_UNC'}.get(args.method,'IQL_BELLMAN_FB')
    params={'const_source_weight':.20,'source_budget':.20} if args.method=='CONST' else {}
    import ctypes
    short=ctypes.create_unicode_buffer(32768)
    assert ctypes.windll.kernel32.GetShortPathNameW(str(out),short,len(short))
    sys.argv=['train.py','--policy',policy,'--env',env,'--shift_level','medium','--srctype','medium','--tartype','medium',
              '--mode','3','--seed',str(args.seed),'--max_step',str(args.steps),'--dir',short.value,'--save-model','--params',json.dumps(params)]
    runpy.run_path(str(ROOT/'train.py'),run_name='__main__')
    write(out/'completed.json',dict(steps=args.steps,method=args.method,seed=args.seed,env=env,ended=time.time()))
if __name__=='__main__':main()
