"""Isolated IQL arms: identical mixed updates, only source Q labels may differ."""
from support_v3 import read,sha
from pathlib import Path
import numpy as np,torch
from algo.utils import ReplayBuffer
from offline_offline.iql import IQL
from offline_offline.iql_bellman_fb import IQL_BELLMAN_FB

METHODS=('TARGET','MIX-CONST','MIX-ORACLE-BACKUP')
ALIASES={'MIX-NAIVE':'MIX-CONST'}

class IndexedReplayBuffer(ReplayBuffer):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs);self.draw_count=0

    def sample(self,batch_size):
        self.last_indices=np.random.randint(0,self.size,size=batch_size)
        self.draw_count+=batch_size
        return tuple(torch.as_tensor(a[self.last_indices],dtype=torch.float32,device=self.device)
                     for a in [self.state,self.action,self.next_state,self.reward,self.not_done])

class SourceBackup:
    """Given source buffer indices, return (next_state, reward, not_done).

    Cache row i corresponds exactly to retained D4RL source row i. Arrays stay
    on CPU; only the selected 64 rows move to the training device.
    """
    def __init__(self,device,cache_receipt=None):
        self.device=device;self.buffer=None;self.arrays=None
        if cache_receipt is not None:
            meta=read(cache_receipt);assert meta['complete'] and not meta['failed_raw_rows']
            assert meta['loader']=='explicit_next_v3'
            self.arrays={}
            for name,record in meta['arrays'].items():
                assert sha(record['path'])==record['sha256']
                self.arrays[name]=np.load(record['path'],mmap_mode='r')
                assert len(self.arrays[name])==meta['n']

    def __call__(self,indices):
        if self.arrays is None:
            assert self.buffer is not None
            arrays=[self.buffer.next_state,self.buffer.reward,self.buffer.not_done]
        else:arrays=[self.arrays[k] for k in ['next_state','reward','not_done']]
        return tuple(torch.as_tensor(a[indices],dtype=torch.float32,device=self.device) for a in arrays)

class TargetIQL(IQL):
    """External batch_size means total draws, compensating for original IQL's x2."""
    def train(self,src,tar,batch_size=128,writer=None):
        assert batch_size==128 and self.config.get('target_only')
        return super().train(src,tar,batch_size//2,writer)

class MixedIQL(IQL_BELLMAN_FB):
    def __init__(self,config,device,source_backup):
        super().__init__(config,device)
        assert self.source_budget==.20 and self.source_loss_weight==1.
        assert not config.get('target_only',False)
        self.source_backup=source_backup;self._q_calls=0;self.source_buffer=None

    def initialize_target_feature_bank(self,tar):
        # Constant weights require neither a kNN bank nor a Bellman reference.
        self.target_features_norm=torch.empty((0,0),device=self.device)

    def compute_source_weight(self,s,a,ns,r,nd):
        self._q_calls=0
        return dict(weight=torch.full_like(r,.20))

    def log_weight_stats(self,stats,writer):
        if writer is not None and (self.total_it==1 or self.total_it%5000==0):
            w=stats['weight'];ess=w.sum().square()/w.square().sum()
            writer.add_scalar('train/source_weight_mean',w.mean(),self.total_it)
            writer.add_scalar('train/source_weight_ess',ess,self.total_it)
            writer.add_scalar('train/source_weight_ess_ratio',ess/w.numel(),self.total_it)

    def q_loss_per_sample(self,state,action,reward,next_state,not_done):
        self._q_calls+=1
        if self._q_calls==1:
            next_state,reward,not_done=self.source_backup(self.source_buffer.last_indices)
        elif self._q_calls!=2:raise RuntimeError('Original mixed IQL source/target Q-call order changed')
        return super().q_loss_per_sample(state,action,reward,next_state,not_done)

    def train(self,src,tar,batch_size=128,writer=None):
        assert batch_size==128
        self.source_buffer=src;self.source_backup.buffer=src
        result=super().train(src,tar,batch_size//2,writer)
        assert self._q_calls==2
        return result

def build(method,config,device,oracle_receipt=None):
    method=ALIASES.get(method,method)
    if method not in METHODS:raise ValueError(method)
    if method=='TARGET':return TargetIQL(config,device)
    if method=='MIX-ORACLE-BACKUP' and oracle_receipt is None:raise ValueError('Missing validated source oracle')
    provider=SourceBackup(device,oracle_receipt if method=='MIX-ORACLE-BACKUP' else None)
    return MixedIQL(config,device,provider)
