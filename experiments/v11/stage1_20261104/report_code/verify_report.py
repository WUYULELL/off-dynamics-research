# -*- coding: utf-8 -*-
from report_runtime import *
import torch
guard();assert not torch.cuda.is_initialized()
m=load('_independent_cpu_verify',HERE/'stage1_verify_repaired.py');m.main()
result=base.r.read(REPORT/'stage1/STAGE1_RESULT.json')
assert result['correction']==base.settings()['actor_loss']
v=base.r.read(REPORT/'SEED207_VECTOR_AUDIT.json')
assert v['nonzero_rows_bitwise_equal_count']==236 and v['nonzero_rows_unequal_count']==104
assert v['all_104_row_differences_are_zero_filled_old_cells'] and v['mismatching_old_nonzero_cells_total']==0
base.r.write(REPORT/'stage1/STAGE1_COMPLETED.json',dict(passed=True,status='AWAITING_USER_DECISION',logical_seed_count=10,
    new_training_runs_in_report_repair=0,new_optimizer_updates=0,new_environment_steps=0,CUDA_initialized=False,
    report_sha256=base.r.sha(REPORT/'stage1/STAGE1_REPORT_ZH.md'),verification_sha256=base.r.sha(REPORT/'stage1/DELIVERY_VERIFIED.json'),
    SAtop_formal_started=False,stage2_started=False,time=time.time()))
print(json.dumps(dict(passed=True,raw_records=70,audits=10,stopped=True),ensure_ascii=False))
