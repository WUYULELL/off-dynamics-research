"""Isolated oracle controls. No edits to project algorithm/registry files."""
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parent.parent
sys.path.insert(0,str(ROOT/'algo'));sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from algo.utils import ReplayBuffer
from offline_offline.iql_bellman_fb import IQL_BELLMAN_FB


class IndexedReplayBuffer(ReplayBuffer):
    """Same NumPy sampler and tensors; retain this particular draw's indices."""
    def sample(self,batch_size):
        self.last_indices=np.random.randint(0,self.size,size=batch_size)
        ids=self.last_indices
        return tuple(torch.FloatTensor(array[ids]).to(self.device) for array in
                     [self.state,self.action,self.next_state,self.reward,self.not_done])


class IQL_BELLMAN_ORACLE(IQL_BELLMAN_FB):
    def __init__(self,config,device,oracle):
        super().__init__(config,device)
        self.oracle=oracle
        self.source_buffer=None

    def initialize_target_feature_bank(self,tar_replay_buffer):
        self.target_features_norm=torch.empty((0,0),device=self.device)

    def train(self,src_replay_buffer,tar_replay_buffer,batch_size=128,writer=None):
        self.source_buffer=src_replay_buffer
        return super().train(src_replay_buffer,tar_replay_buffer,batch_size,writer)

    def oracle_tensors(self):
        ids=self.source_buffer.last_indices
        return tuple(torch.as_tensor(self.oracle[key][ids],dtype=torch.float32,device=self.device)
                     for key in ['oracle_next','oracle_reward','oracle_not_done'])

    def compute_source_weight(self,s,a,ns,r,nd):
        on,orr,ond=self.oracle_tensors()
        with torch.no_grad():
            ys=r+nd*self.discount*self.v_func(ns)
            yo=orr.reshape(-1,1)+ond.reshape(-1,1)*self.discount*self.v_func(on)
            b=torch.abs(ys-yo)
            scaled=b/(b.mean().detach()+1e-6) if self.normalize_bellman else b
            raw=torch.exp(-self.bellman_eta*scaled)
            w=(self.source_budget*raw/(raw.mean().detach()+1e-6)).clamp(self.min_weight,self.max_weight)
        return dict(weight=w.detach(),bellman_mismatch=b.detach(),bellman_weight_raw=raw.detach(),
                    knn_sa_distance=torch.zeros_like(b))


class IQL_ORACLE_BACKUP(IQL_BELLMAN_ORACLE):
    """Constant .20; only source Q backup uses oracle transition and done mask.

    This is an ideal-transition ablation, not a theorem about maximal return.
    """
    def compute_source_weight(self,s,a,ns,r,nd):
        self._q_calls=0
        w=torch.full_like(r,self.source_budget)
        return dict(weight=w,bellman_mismatch=torch.zeros_like(r),
                    bellman_weight_raw=torch.ones_like(r),knn_sa_distance=torch.zeros_like(r))

    def q_loss_per_sample(self,state,action,reward,next_state,not_done):
        self._q_calls+=1
        if self._q_calls==1:
            next_state,reward,not_done=self.oracle_tensors()
            reward=reward.reshape(-1,1);not_done=not_done.reshape(-1,1)
        elif self._q_calls!=2:
            raise RuntimeError('Unexpected parent Q update order')
        return super().q_loss_per_sample(state,action,reward,next_state,not_done)
