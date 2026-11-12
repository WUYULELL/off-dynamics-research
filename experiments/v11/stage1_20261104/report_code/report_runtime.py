# -*- coding: utf-8 -*-
"""CPU report-only routing; imports accepted readers, never calls training entry points."""
import sys,os,json,time,hashlib,importlib.util
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='-1';sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE_CODE=ROOT/'oracle_diag/v11_saall_finish_20261103'
sys.path.insert(0,str(BASE_CODE))
import common as base
REPORT=ROOT/'outputs/v11_saall_report_20261104'
def guard():
    base.guard()
    for p,h in base.r.read(REPORT/'REPORT_CODE_FROZEN.json')['sha256'].items():assert base.r.sha(p)==h,p
def protect(label):
    manifest=base.r.read(REPORT/'protected_manifest.json')['sha256']
    changed=[p for p,h in manifest.items() if not Path(base.r.winpath(p)).is_file() or base.r.sha(p)!=h]
    receipt=dict(passed=not changed,count=len(manifest),changed=changed,time=time.time())
    base.r.write(REPORT/('protected_'+label+'.json'),receipt);assert not changed,changed[:3]
    return receipt
def load(name,file):
    spec=importlib.util.spec_from_file_location(name,file);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    m.OUT=REPORT;m.STAGE1=REPORT/'stage1';m.guard=guard;m.protect=protect
    return m
