"""Independent CPU check of zero ranges, partial rows and exact JSONL bytes."""
import json,hashlib
from pathlib import Path
import numpy as np
P=Path(__file__).resolve().parent;ROOT=P.parents[1]
read=lambda p:json.loads(p.read_text(encoding='utf-8-sig'))
v=read(P/'SEED207_VECTOR_AUDIT.json');paths=list(v['file_sha256'])
a=np.load(paths[0],mmap_mode='r');b=np.load(paths[1],mmap_mode='r')
mask=np.zeros(500000,dtype=bool);equal_count=0;partial_count=0
for start in range(0,500000,10000):
    x=a[start:start+10000];mask[start:start+len(x)]=np.count_nonzero(x,axis=(1,2))==0
    if start>=448506:continue
    end=min(start+10000,448506);x=a[start:end];y=b[start:end]
    nz=np.count_nonzero(x,axis=(1,2))>0
    neq=x.view(np.uint32)!=y.view(np.uint32)
    eq=~np.any(neq,axis=(1,2));equal_count+=int(np.count_nonzero(nz & eq));partial_count+=int(np.count_nonzero(nz & ~eq))
    assert not np.any(neq & (x!=0))
expected=np.zeros(500000,dtype=bool)
for first,last in v['zero_ranges_full']:expected[first-1:last]=True
assert np.array_equal(expected,mask)
assert int(mask.sum())==499660 and int(mask[:448506].sum())==448166
assert equal_count==236 and partial_count==104
old=ROOT/'outputs/v11_actor_w64_saall_20261102/stage1/runs/hopp_SAall_s207/steps.jsonl'
new=ROOT/'outputs/v11_saall_finish_20261103/stage1/runs/seed207_rerun/steps.jsonl'
h=hashlib.sha256()
with open(old,'rb') as x,open(new,'rb') as y:
    for step in range(1,448507):
        p,q=x.readline(),y.readline();assert p and p==q;h.update(p)
    assert x.read(1)==b''
assert h.hexdigest()==v['comparison_log']['old_prefix_sha256']
result=dict(passed=True,full_zero_rows=int(mask.sum()),prefix_zero_rows=int(mask[:448506].sum()),
    nonzero_rows=equal_count+partial_count,fully_equal_nonzero_rows=equal_count,partial_nonzero_rows=partial_count,
    all_zero_ranges_independently_verified=True,all_old_nonzero_values_bitwise_match=True,
    requested_JSONL_steps=448506,requested_JSONL_byte_equal=True,prefix_sha256=h.hexdigest(),
    new_optimizer_updates=0,new_environment_steps=0)
with open(P/'INDEPENDENT_VECTOR_VERIFIED.json','x',encoding='utf-8') as f:json.dump(result,f,ensure_ascii=False,indent=2)
print(json.dumps(result))
