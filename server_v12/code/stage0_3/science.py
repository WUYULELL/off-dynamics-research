"""Original R14 T and V9 BC-all definitions, extracted without old queue startup."""
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
        self.items={x['path']:x for x in read(self.root/'MANIFEST.json')['files']}
        self.specification=read(self.path('cluster/paths.example.json'))['tasks']['hopper-morph']
        self.expected=read(HERE/'SCIENCE_AST.json');self.science={}

    def path(self,relative):
        p=(self.root/relative).resolve();x=self.items[relative]
        assert self.root in p.parents and p.is_file()
        assert p.stat().st_size==x['bytes'] and sha(p)==x['sha256'],'Scientific input changed: '+relative
        return p

    def task_spec(self,task):
        assert task=='hopper-morph'
        s=dict(self.specification)
        for k in ('target_dataset','xml'):
            p=self.path(s[k]);s[k]=str(p);s[k+'_sha256']=sha(p)
        return s

    def extract(self,relative,names,namespace):
        p=self.path(relative);tree=ast.parse(p.read_text(encoding='utf-8-sig'))
        selected=[]
        for name in names:
            nodes=[n for n in tree.body if isinstance(n,(ast.ClassDef,ast.FunctionDef)) and n.name==name]
            assert len(nodes)==1
            n=nodes[0];h=hashlib.sha256(ast.dump(n,include_attributes=False).encode()).hexdigest()
            assert self.expected[relative+'::'+name]==h
            self.science[relative+'::'+name]=h;selected.append(n)
        code=ast.Module(body=selected,type_ignores=[]);ast.fix_missing_locations(code)
        exec(compile(code,str(p),'exec'),namespace)

    def load(self):
        import sys
        import numpy as np
        import torch
        assert not torch.cuda.is_initialized()
        for f in ('iql_dyn.py','iql_unc.py'):self.path('source_snapshot/algo/offline_offline/'+f)
        algo=self.root/'source_snapshot/algo';sys.path.insert(0,str(algo))
        mod=importlib.import_module('offline_offline.iql_dyn')
        unc=importlib.import_module('offline_offline.iql_unc')
        assert algo in Path(mod.__file__).resolve().parents and algo in Path(unc.__file__).resolve().parents
        r=types.ModuleType('_v12_03_R14')
        r.__dict__.update(hashlib=hashlib,sha=sha,task_spec=self.task_spec)
        self.extract('source_snapshot/oracle_diag/coverage_raw_r14/runtime.py',['seed','model_config','target_data'],r.__dict__)
        iq=dict(np=np,torch=torch,hashlib=hashlib,IQL_DYN=mod.IQL_DYN,seed=r.seed)
        self.extract('source_snapshot/oracle_diag/coverage_raw_r14/learner.py',['Learner','Sampler','initialize'],iq)
        ie=dict(np=np,torch=torch,random=random,task_spec=self.task_spec)
        self.extract('source_snapshot/oracle_diag/coverage_raw_r14/train.py',['evaluate'],ie)
        bc_runtime=types.ModuleType('_v12_03_V9_BC')
        selection=self.path('records/outputs/v9_hopper_label_headroom_20261031/stage3/selections.json')
        def read_selection(path):
            assert Path(path).name=='selections.json'
            return read(selection)
        bc_runtime.__dict__.update(hashlib=hashlib,read=read_selection,OUT=Path('/_v12_readonly_selection_route'))
        self.extract('source_snapshot/oracle_diag/v9_hopper_label_headroom/stage3/runtime.py',['seed','ids'],bc_runtime.__dict__)
        bc=dict(np=np,torch=torch,random=random,hashlib=hashlib,Policy=unc.Policy,
                seed=bc_runtime.seed,ids=bc_runtime.ids,task_spec=self.task_spec)
        self.extract('source_snapshot/oracle_diag/v9_hopper_label_headroom/stage3/learner.py',['Actor','Sampler'],bc)
        be=dict(np=np,torch=torch,random=random,task_spec=self.task_spec)
        self.extract('source_snapshot/oracle_diag/v9_hopper_label_headroom/stage3/train.py',['evaluate'],be)
        self.r,self.iql,self.bc,self.bc_runtime=r,iq,bc,bc_runtime
        self.iql_evaluate,self.bc_evaluate=ie['evaluate'],be['evaluate']
        assert self.science==self.expected and not torch.cuda.is_initialized()

    def iql_config(self,seed):
        assert 901<=seed<=906
        c=self.r.model_config(10.,500000,seed,'hopper-morph',0.)
        baseline=read(self.path('records/oracle_diag/results/coverage_raw_20261026_r14/runs/hopp_w0_s201/config.json'))['learner']
        baseline=dict(baseline,seed=seed)
        assert c==baseline
        return c
