"""Isolated two-step synthetic CPU equivalence probe for either old or new learner."""
import argparse
import hashlib
from pathlib import Path
import sys

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--code',required=True);parser.add_argument('--w',type=float,required=True);parser.add_argument('--out',required=True);a=parser.parse_args()
    sys.dont_write_bytecode=True;sys.path.insert(0,a.code)
    import numpy as np
    import torch
    from runtime import configure_cpu,model_config,write
    from learner import initialize
    configure_cpu();model=initialize(model_config(10,500000,201,'hopper-morph',a.w),'cpu')
    init=model.digest();rng=np.random.RandomState(20261030);losses=[]
    def batch():
        return tuple(torch.tensor(x,dtype=torch.float32) for x in (rng.randn(64,11),rng.randn(64,3),rng.randn(64,11),rng.randn(64,1),np.ones((64,1))))
    for _ in range(2):losses.append([float(x) for x in model.step(batch(),batch())])
    h=hashlib.sha256()
    def visit(x):
        if torch.is_tensor(x):h.update(str(x.dtype).encode());h.update(str(tuple(x.shape)).encode());h.update(x.numpy().tobytes())
        elif isinstance(x,dict):
            for key in sorted(x,key=str):h.update(str(key).encode());visit(x[key])
        elif isinstance(x,(list,tuple)):
            for value in x:visit(value)
        else:h.update(repr(x).encode())
    visit(model.checkpoint());assert model.finite() and not torch.cuda.is_initialized()
    write(Path(a.out),dict(initialization=init,losses=losses,complete_state_digest=h.hexdigest(),model_digest=model.digest(),
        synthetic_updates=2,real_data_updates=0,environment_steps=0,CUDA_initialized=False))

if __name__=='__main__':main()
