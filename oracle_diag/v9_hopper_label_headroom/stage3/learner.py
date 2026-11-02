from runtime import *
import random
import numpy as np
import torch
from offline_offline.iql_unc import Policy
class Actor:
    def __init__(self,job,device):
        self.job=job;self.device=device;self.total_it=0
        s=seed('initialization',job['seed'],job['task']);random.seed(s);np.random.seed(s);torch.manual_seed(s)
        if device!='cpu':torch.cuda.manual_seed_all(s)
        spec=task_spec(job['task'])
        self.policy=Policy(spec['state_dim'],spec['action_dim'],1.,256).to(device)
        self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=3e-4)
        self.scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer,200000)
    def step(self,state,action):
        self.total_it+=1
        loss=self.policy.bc_loss(state,action).mean(dim=1,keepdim=True).mean()
        self.optimizer.zero_grad();loss.backward();self.optimizer.step();self.scheduler.step()
        return loss.detach()
    def deterministic_action(self,state):
        with torch.no_grad():
            x=torch.as_tensor(state,dtype=torch.float32,device=self.device).reshape(1,-1)
            return torch.tanh(self.policy.network(x).chunk(2,dim=1)[0]).squeeze(0).cpu().numpy()
    def digest(self):
        return hashlib.sha256(b''.join(p.detach().cpu().numpy().tobytes() for p in self.policy.state_dict().values())).hexdigest()
    def finite(self):return all(bool(torch.isfinite(p).all()) for p in self.policy.parameters())
    def checkpoint(self):return dict(policy=self.policy.state_dict(),actor_optimizer=self.optimizer.state_dict(),actor_scheduler=self.scheduler.state_dict(),step=self.total_it,job=self.job)
class Sampler:
    def __init__(self,arrays,job,device):
        self.arrays=arrays;self.job=job;self.device=device;self.ids=np.asarray(ids(job),dtype=np.int64)
        self.rng=np.random.RandomState(seed('target_sample',job['seed'],job['task']));self.hash=hashlib.sha256();self.draws=0
    def sample(self):
        ix=self.ids[self.rng.randint(len(self.ids),size=64)];self.hash.update(ix.astype('<i8').tobytes());self.draws+=64
        return tuple(torch.as_tensor(np.asarray(a[ix]),dtype=torch.float32,device=self.device) for a in self.arrays[:2])
