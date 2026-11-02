"""Keep the v4 update order; remove only the specified direct source loss."""
from support_v5 import *
from methods_v4 import OracleW1IQL
from methods_v3 import SourceBackup, IndexedReplayBuffer

class PathIQL(OracleW1IQL):
    def __init__(self,config,device,source_backup,method):
        super().__init__(config,device,source_backup)
        assert method in A_METHODS
        self.path_method=method
        self._loss_calls=0

    def compute_source_weight(self,*args):
        self._loss_calls=0
        return super().compute_source_weight(*args)

    def combine_target_source_loss(self,target_loss,source_loss,source_weight):
        stage=('V','Q','actor')[self._loss_calls]
        self._loss_calls+=1
        if stage=='actor' or (stage=='V' and self.path_method=='Q-ORACLE'):
            return target_loss.mean()
        return super().combine_target_source_loss(target_loss,source_loss,source_weight)

    def train(self,*args,**kwargs):
        result=super().train(*args,**kwargs)
        assert self._loss_calls==3
        return result

def build(method,config,device,receipt=None):
    assert method in A_METHODS+('ORACLE-W1',)
    assert receipt is not None or method=='CRITIC-SOURCE'
    provider=SourceBackup(device,None if method=='CRITIC-SOURCE' else receipt)
    if method=='ORACLE-W1':return OracleW1IQL(config,device,provider)
    return PathIQL(config,device,provider,method)
