# -*- coding: utf-8 -*-
"""Create a new stage2 implementation; never alter R14 or v9 stage0/1."""
import ast
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'oracle_diag/coverage_raw_r14_finish1'
DEST=HERE/'stage2'

def replace_function(source,name,replacement):
    node=next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    lines=source.splitlines(keepends=True)
    return ''.join(lines[:node.lineno-1])+replacement+'\n'+''.join(lines[node.end_lineno:])

def main():
    DEST.mkdir(exist_ok=False)
    runtime=(OLD/'runtime.py').read_text(encoding='utf-8')
    runtime=runtime.replace('ROOT = HERE.parents[1]','ROOT = HERE.parents[2]')
    runtime=runtime.replace("OUT = ROOT / 'oracle_diag/results/coverage_raw_20261029_r14_finish1'","OUT = ROOT / 'outputs/v9_hopper_label_headroom_20261030/stage2'")
    runtime=runtime.replace("DOC = ROOT / 'outputs/research_idea_protocol_20261029_r14_finish1'","DOC = OUT")
    runtime=runtime.replace("os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'","os.environ['D4RL_SUPPRESS_IMPORT_ERROR'] = '1'\nos.environ.pop('ODRL_PAPER_STYLE_SOURCE_TARGET', None)")
    runtime=runtime.replace("with open(tmp, 'w', encoding='utf-8') as f:","with open(winpath(tmp), 'w', encoding='utf-8') as f:")
    runtime=runtime.replace('os.replace(str(tmp), str(path))','os.replace(winpath(tmp), winpath(path))')
    runtime=replace_function(runtime,'check_frozen',r'''def check_frozen():
    auth=read(OUT/'stage_authorization.json')
    assert auth['scope']=='V9_stage2_hopper_label_attribution_20_only'
    assert auth['new_GPU_jobs']==20 and not auth['estimator_development_authorized']
    assert not auth['stage3_or_4_authorized']
    assert sha(OUT/'protocol.json')==auth['protocol_sha256']
    for name in ('code_frozen.json','dependency_hashes.json'):
        for p,h in read(OUT/name)['sha256'].items():assert sha(p)==h,'Frozen file changed: '+p
''')
    runtime=replace_function(runtime,'validate_spec',r'''def validate_spec(job):
    assert job['task']=='hopper-morph' and job['phase']=='label_attribution'
    assert job['arm']=='M' and job['source_w'] in (.25,1.)
    assert job['source_lambda']==job['source_w']/(1.+job['source_w'])
    assert job['source_labels']=='oracle_target_dynamics'
    assert 201<=job['seed']<=210 and job['steps']==500000 and job['temp']==10
    assert Path(job['out']).resolve().parent==OUT/'runs'
''')
    runtime += r'''

CACHE_STAGE1=ROOT/'outputs/v9_hopper_label_headroom_20261030/stage1'
CACHE_RECEIPT=CACHE_STAGE1/'STAGE1_COMPLETED.json'
raw_source_data_R14=source_data

def source_data(task,verify=True):
    import numpy as np
    arrays,ids=raw_source_data_R14(task,verify)
    receipt=read(CACHE_RECEIPT)
    assert receipt['passed'] and receipt['canonical_rows']==999998 and receipt['independent_acceptance_passed']
    assert np.array_equal(ids,np.load(winpath(CACHE_STAGE1/'cache/raw_indices.npy')))
    for ix,key,shape in ((2,'next_state',(999998,11)),(3,'reward',(999998,1)),(4,'not_done',(999998,1))):
        item=receipt['cache_arrays'][key]
        if verify:assert sha(item['path'])==item['sha256']
        value=np.load(winpath(item['path']),mmap_mode='r',allow_pickle=False)
        assert value.shape==shape and value.dtype==np.float32 and np.isfinite(value).all()
        arrays[ix]=value
    assert np.isin(arrays[4],[0.,1.]).all()
    return arrays,ids
'''
    (DEST/'runtime.py').write_text(runtime,encoding='utf-8')
    (DEST/'learner.py').write_bytes((OLD/'learner.py').read_bytes())
    train=(OLD/'train.py').read_text(encoding='utf-8')
    assert train.count("source_labels='original_dataset',oracle_loaded=False,")==1
    train=train.replace("source_labels='original_dataset',oracle_loaded=False,","source_labels='oracle_target_dynamics',oracle_loaded=True,oracle_stage1_receipt=str(CACHE_RECEIPT),oracle_stage1_receipt_sha256=sha(CACHE_RECEIPT),")
    (DEST/'train.py').write_text(train,encoding='utf-8')
    audit=(OLD/'audit.py').read_text(encoding='utf-8')
    audit=audit.replace("alignment['source_labels']=='original_dataset' and alignment['oracle_loaded'] is False","alignment['source_labels']=='oracle_target_dynamics' and alignment['oracle_loaded'] is True\n        assert alignment['oracle_stage1_receipt_sha256']==sha(CACHE_RECEIPT)")
    (DEST/'audit.py').write_text(audit,encoding='utf-8')
    print('Created new stage2 runtime/learner/train/audit; original files untouched')

if __name__=='__main__':main()
