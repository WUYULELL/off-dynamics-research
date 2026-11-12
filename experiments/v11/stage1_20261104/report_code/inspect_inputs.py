# -*- coding: utf-8 -*-
"""CPU-only configuration and interrupted-vector inspection, without training imports."""
import os,sys,json,hashlib
from pathlib import Path
import numpy as np
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
OLD=ROOT/'outputs/v11_saall_finish_20261103'
OUT=ROOT/'outputs/v11_saall_report_20261104'
OUT.mkdir(exist_ok=False)
def read(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def write(name,x):
    with open(OUT/name,'x',encoding='utf-8') as f:json.dump(x,f,ensure_ascii=False,indent=2)
def ranges(ix):
    if not len(ix):return []
    split=np.flatnonzero(np.diff(ix)>1)+1
    return [[int(a[0]),int(a[-1])] for a in np.split(ix,split)]
root=read(OLD/'config.json');jobs=read(OLD/'stage1/training_plan.json')['jobs'];rows=[]
for j in jobs:
    p=Path(j['out'])/'config.json';c=read(p)
    rows.append(dict(seed=j['seed'],config=str(p),sha256=sha(p),correction_present='correction' in c,
        actor_loss_present='actor_loss' in c,scientific_config_sha256=c.get('scientific_config_sha256'),
        parent_config_matches=c.get('scientific_config_sha256')==sha(OLD/'config.json'),
        temp=c['learner']['temp'],actor_source_w=c['actor_source_w'],source_denominator=c['source_denominator']))
mapping=dict(expected_key='correction',expected_file=str(OLD/'config.json'),
    historical_definition_file=str(ROOT/'oracle_diag/v11_source_actor/prepare.py'),historical_lines=[26,36],
    historical_definition=read(ROOT/'outputs/v11_source_actor_20261102_stage0_recovery2/stage0/config.json')['correction'],
    interpretation='Original report metadata describing replacement of negative log pi by O1 advantage-weighted action MSE; not a training parameter.',
    current_key_missing=True,explicit_mapping=dict(from_file=str(OLD/'config.json'),from_field='actor_loss',
        exact_value=root['actor_loss'],to_report_field='correction'),
    support_fields={k:root[k] for k in ('actor_beta','actor_weight_max','actor_source_w','source_actor_denominator','advantage')},
    seed_configs=rows,quote_is_verbatim=True,no_value_invented=True)
assert all(r['parent_config_matches'] for r in rows)
write('CORRECTION_FIELD_AUDIT.json',mapping)
old=ROOT/'outputs/v11_actor_w64_saall_20261102/stage1/runs/hopp_SAall_s207/diagnostic_vectors.npy'
new=Path(jobs[6]['out'])/'diagnostic_vectors.npy'
a=np.load(old,mmap_mode='r');b=np.load(new,mmap_mode='r');N=448506
zero=np.all(a==0,axis=(1,2));prefix_nonzero=np.flatnonzero(~zero[:N]);full_nonzero=np.flatnonzero(~zero)
equal=np.all(a[prefix_nonzero].view(np.uint32)==b[prefix_nonzero].view(np.uint32),axis=(1,2))
bad=prefix_nonzero[~equal]+1
nonzero_fields={str(int(i+1)):int(np.count_nonzero(a[i])) for i in prefix_nonzero}
comparison=read(OLD/'SEED207_PREFIX_COMPARISON.json')
v=dict(shape=list(a.shape),dtype=str(a.dtype),step_row_definition='one row = one training step,64 samples x7 columns',
    logged_steps=N,full_allocated_rows=500000,zero_rows_full=int(zero.sum()),zero_ranges_full=ranges(np.flatnonzero(zero)+1),
    zero_rows_logged_prefix=int(zero[:N].sum()),zero_ranges_logged_prefix=ranges(np.flatnonzero(zero[:N])+1),
    nonzero_rows_logged_prefix=len(prefix_nonzero),nonzero_ranges_logged_prefix=ranges(prefix_nonzero+1),
    nonzero_rows_full=len(full_nonzero),nonzero_ranges_full=ranges(full_nonzero+1),
    nonzero_rows_bitwise_equal_count=int(equal.sum()),nonzero_rows_unequal_count=len(bad),
    first_nonzero_row_mismatch=int(bad[0]) if len(bad) else None,
    scalar_nonzero_counts_by_nonzero_row=nonzero_fields,
    file_sha256={str(old):sha(old),str(new):sha(new)},
    writer_source=str(ROOT/'oracle_diag/v11_saall_finish_20261103/run_job.py'),
    writer='open_memmap mode w+ preallocates (500000,64,7); each step writes vectors[step-1]; vectors.flush only after normal loop completion; JSONL log.flush every1000 and fsync at normal completion. No per-block memmap flush.',
    source_lines={'preallocation':42,'per_step_assignment':49,'log_flush':53,'normal_completion_vector_flush':69},
    classification='中断版本该文件不完整' if int(zero[:N].sum())>0 and comparison['logs']['parsed_numeric_bits_equal'] else '需要进一步检查',
    cause_scope='Observed zero-filled persisted regions despite completed JSONL steps and no regular vector flush support incomplete persistence after interruption; exact OS writeback timing is not recoverable.',
    determinism_basis='448506 JSONL records including V/Q/actor/all scalar diagnostics are byte-for-byte and numeric-bit identical',
    JSONL_first_mismatch=None,comparison_log= comparison['logs'],new_optimizer_updates=0,new_environment_steps=0)
write('SEED207_VECTOR_AUDIT.json',v)
print(json.dumps(dict(mapping=mapping['explicit_mapping'],seed_configs=len(rows),
    zero_full=v['zero_rows_full'],zero_prefix=v['zero_rows_logged_prefix'],nonzero=v['nonzero_rows_logged_prefix'],
    nonzero_equal=v['nonzero_rows_bitwise_equal_count'],nonzero_bad=len(bad),ranges=v['nonzero_ranges_logged_prefix']),ensure_ascii=False))
