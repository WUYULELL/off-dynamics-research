"""Unchanged R14 T definitions with Linux paths; never import the old queue runtime."""
import ast
import hashlib
import importlib
from pathlib import Path
import random
import types
from common import HERE,read,sha


class Science:
    def __init__(self,root):
        self.root=Path(root).resolve()
        self.manifest={x['path']:x for x in read(self.root/'MANIFEST.json')['files']}
        self.specification=read(self.path('cluster/paths.example.json'))['tasks']['hopper-morph']
        self.expected=read(HERE/'SCIENCE_AST.json')
        self.science={}
        self.checked={}

    def path(self,rel):
        p=(self.root/rel).resolve()
        assert self.root in p.parents and p.is_file()
        x=self.manifest[rel]
        assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'], 'Changed science input: '+rel
        return p

    def task_spec(self,task):
        assert task=='hopper-morph','Only the authorized Hopper T acceptance is supported'
        s=dict(self.specification)
        for k in ('target_dataset','xml'):
            p=self.path(s[k]);s[k]=str(p);s[k+'_sha256']=sha(p)
        return s

    def extract(self,rel,names,namespace):
        p=self.path(rel)
        tree=ast.parse(p.read_text(encoding='utf-8-sig'),filename=str(p))
        selected=[]
        for name in names:
            nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name]
            assert len(nodes)==1,(rel,name)
            n=nodes[0]
            h=hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
            assert self.expected[rel+'::'+name]==h,'Changed scientific AST'
            self.science[rel+'::'+name]=h
            selected.append(n)
        module=ast.Module(body=selected,type_ignores=[])
        ast.fix_missing_locations(module)
        exec(compile(module,str(p),'exec'),namespace)

    def load(self):
        import sys
        import numpy as np
        import torch
        assert not torch.cuda.is_initialized()
        for name in ('iql_dyn.py','iql_unc.py'):
            self.path('source_snapshot/algo/offline_offline/'+name)
        algo=self.root/'source_snapshot/algo'
        sys.path.insert(0,str(algo))
        module=importlib.import_module('offline_offline.iql_dyn')
        for name in ('offline_offline.iql_dyn','offline_offline.iql_unc'):
            assert algo in Path(sys.modules[name].__file__).resolve().parents
        runtime=types.ModuleType('_v12_R14_T_runtime')
        runtime.__dict__.update(sha=sha,task_spec=self.task_spec,hashlib=hashlib)
        self.extract('source_snapshot/oracle_diag/coverage_raw_r14/runtime.py',
                     ['seed','model_config','target_data'],runtime.__dict__)
        ns=dict(__name__='_v12_R14_T_science',np=np,torch=torch,hashlib=hashlib,
                IQL_DYN=module.IQL_DYN,seed=runtime.seed)
        self.extract('source_snapshot/oracle_diag/coverage_raw_r14/learner.py',
                     ['Learner','Sampler','initialize'],ns)
        ev=dict(np=np,torch=torch,random=random,task_spec=self.task_spec)
        self.extract('source_snapshot/oracle_diag/coverage_raw_r14/train.py',['evaluate'],ev)
        self.runtime,self.learner,self.evaluate=runtime,ns,ev['evaluate']
        assert self.science==self.expected
        assert not torch.cuda.is_initialized()

    def config(self):
        result=self.runtime.model_config(10.,500000,201,'hopper-morph',0.)
        baseline=read(self.path('records/oracle_diag/results/coverage_raw_20261026_r14/runs/hopp_w0_s201/config.json'))
        assert result==baseline['learner'],'Hopper T parameters differ from R14'
        assert baseline['batch_target']==64 and baseline['batch_source']==0
        assert not any(baseline[x] for x in ('V_source','Q_source','actor_source'))
        return result
