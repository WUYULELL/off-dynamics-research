"""Instrument the existing train.py in memory, leaving every old file intact."""
import os,sys,json,argparse,runpy,time,importlib,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
for path in [ROOT,ROOT/'algo']:sys.path.insert(0,str(path))
os.environ['D4RL_SUPPRESS_IMPORT_ERROR']='1'
mu=Path.home()/'.mujoco210/mujoco210';os.environ['MUJOCO_PY_MUJOCO_PATH']=str(mu)
os.environ['PATH']=str(mu/'bin')+os.pathsep+os.environ.get('PATH','')


def main():
    p=argparse.ArgumentParser();p.add_argument('--env',required=True)
    p.add_argument('--method',choices=['TARGET','MIX_FB','CONST','ORACLE','BACKUP'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--steps',type=int,default=500000)
    p.add_argument('--check-eval',type=int,default=0)
    args=p.parse_args();out=args.out.resolve()
    if out.exists() and any(out.iterdir()):raise RuntimeError('Refusing overwrite '+str(out))
    out.mkdir(parents=True,exist_ok=True)
    import torch,numpy as np
    torch.set_num_threads(2)
    if not torch.cuda.is_available():raise RuntimeError('CUDA required for training')
    import algo.utils as utils
    from oracle_methods import IndexedReplayBuffer,IQL_BELLMAN_ORACLE,IQL_ORACLE_BACKUP
    utils.ReplayBuffer=IndexedReplayBuffer
    registry=importlib.import_module('algo.call_algo');original=registry.call_algo
    oracle_path=HERE/'results/verified_20261003/diagnostics'/(args.env+'_source_oracle.npz')
    oracle=dict(np.load(oracle_path))
    def factory(policy_name,config,mode,device):
        if args.check_eval:config['eval_freq']=args.check_eval
        if args.method=='ORACLE':model=IQL_BELLMAN_ORACLE(config,device,oracle)
        elif args.method=='BACKUP':model=IQL_ORACLE_BACKUP(config,device,oracle)
        else:model=original(policy_name,config,mode,device)
        (out/'effective_config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
        base_train=model.train;start=time.time();checked=False
        def instrument(src,tar,batch_size=128,writer=None):
            nonlocal checked
            if not checked:
                for field,key in [('state','source_state'),('action','source_action'),('next_state','source_next'),
                                   ('reward','source_reward'),('not_done','source_not_done')]:
                    assert np.array_equal(getattr(src,field).reshape(oracle[key].shape),oracle[key]),field
                assert config['source_budget']==.2 if 'source_budget' in config else True
                assert config.get('bellman_eta',1)==1 and batch_size==128
                (out/'data_verified.json').write_text(json.dumps(dict(source_n=src.size,target_n=tar.size,
                    aligned=True,oracle_sha256=hashlib.sha256(oracle_path.read_bytes()).hexdigest())),encoding='utf-8')
                checked=True
            base_train(src,tar,batch_size,writer)
            step=model.total_it
            if step%5000==0 or step==args.steps:
                if not all(torch.isfinite(param).all() for net in [model.policy,model.v_func,model.q_funcs] for param in net.parameters()):
                    raise FloatingPointError('Nonfinite network parameters at '+str(step))
                (out/'progress.json').write_text(json.dumps(dict(step=step,elapsed=time.time()-start,pid=os.getpid())),encoding='utf-8')
            if step%100000==0 or step==args.steps:
                target=out/'values';target.mkdir(exist_ok=True)
                torch.save(model.v_func.state_dict(),str(target/('V_'+str(step)+'.pt')))
        model.train=instrument
        return model
    registry.call_algo=factory
    for key in list(os.environ):
        if key.startswith('ODRL_'):del os.environ[key]
    os.environ.update(ODRL_PAPER_STYLE_SOURCE_TARGET='1',ODRL_TARGET_FRACTION='0.1',
        ODRL_SOURCE_BUDGET='0.20',ODRL_BELLMAN_ETA='1.0',ODRL_TARGET_ONLY='1' if args.method=='TARGET' else '0')
    policy={'TARGET':'IQL','CONST':'IQL_UNC'}.get(args.method,'IQL_BELLMAN_FB')
    params={'const_source_weight':.20,'source_budget':.20} if args.method=='CONST' else {}
    directory=str(out)
    if os.name=='nt':
        # Old train.py appends '/models'; the Win32 extended prefix forbids
        # mixed separators. A real NTFS short path keeps the old entry intact.
        import ctypes
        short=ctypes.create_unicode_buffer(32768)
        if not ctypes.windll.kernel32.GetShortPathNameW(str(out),short,len(short)):
            raise OSError('Could not obtain short path for original train.py')
        directory=short.value
    sys.argv=['train.py','--policy',policy,'--env',args.env,'--shift_level','medium','--srctype','medium',
        '--tartype','medium','--mode','3','--seed','0','--max_step',str(args.steps),'--dir',directory,
        '--save-model','--params',json.dumps(params)]
    runpy.run_path(str(ROOT/'train.py'),run_name='__main__')
    (out/'completed.json').write_text(json.dumps(dict(steps=args.steps,method=args.method,env=args.env,ended=time.time())),encoding='utf-8')


if __name__=='__main__':main()
