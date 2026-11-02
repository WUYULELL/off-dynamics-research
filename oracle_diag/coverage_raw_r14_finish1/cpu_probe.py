"""Isolated deterministic synthetic CPU algebra probe; no simulator or CUDA."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

def main():
    p=argparse.ArgumentParser();p.add_argument('--code',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    sys.dont_write_bytecode=True;sys.path.insert(0,args.code)
    import numpy as np
    import torch
    from runtime import configure_cpu,model_config,write
    from learner import initialize
    configure_cpu()
    m=initialize(model_config(10,500000,204,'hopper-morph',1.),'cpu')
    initial=m.digest(); rng=np.random.RandomState(12345);losses=[]
    def batch():
        return tuple(torch.tensor(x,dtype=torch.float32) for x in (rng.randn(64,11),rng.randn(64,3),rng.randn(64,11),rng.randn(64,1),np.ones((64,1))))
    for _ in range(2): losses.append([float(x) for x in m.step(batch(),batch())])
    h=hashlib.sha256()
    def visit(v):
        if torch.is_tensor(v): h.update(str(v.dtype).encode());h.update(str(tuple(v.shape)).encode());h.update(v.cpu().numpy().tobytes())
        elif isinstance(v,dict):
            for k in sorted(v,key=str): h.update(str(k).encode());visit(v[k])
        elif isinstance(v,(list,tuple)):
            for x in v:visit(x)
        else: h.update(repr(v).encode())
    visit(m.checkpoint())
    assert not torch.cuda.is_initialized() and m.finite()
    write(Path(args.out),dict(initialization=initial,losses=losses,model_digest=m.digest(),complete_state_digest=h.hexdigest(),CPU_only=True,CUDA_initialized=False,synthetic_updates=2,real_data_updates=0,environment_steps=0))

if __name__=='__main__':main()
