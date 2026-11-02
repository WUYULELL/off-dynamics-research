from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
import re

source = Path('outputs/学生成绩单_修改版.doc')
target = Path('outputs/学生成绩单_修改版_3.21.doc')
original = source.read_bytes()
start = original.index('贵州大学研究生成绩单'.encode('utf-16-le'))
end = original.index('成绩审核签章：'.encode('utf-16-le'))
text = original[start:end].decode('utf-16-le')
pattern = r'([^\x07\r]+)\x07(\d+)\x07([123]\.0)\x07([是否])\x07(\d{2})\x07(\d\.\d{2})\x07'
rows = list(re.finditer(pattern, text))
assert len(rows) == 12
fixed = {'高级算法设计与分析':76,'大数据平台技术':94,'数字经济与产品思维':94}
def gp(score):
    return next(point for minimum,point in [(90,40),(85,37),(82,33),(78,30),(75,27),(71,23),(66,20),(62,17),(60,13)] if score>=minimum)

# Keep the requested scores fixed; minimize total score movement elsewhere.
states = {0:((0,0),[])}
for row in rows:
    name,hours,credits,degree,score,points = row.groups()
    score=int(score); credits=int(float(credits))
    candidates=[fixed[name]] if name in fixed else range(60,100)
    updated={}
    for total,(cost,chosen) in states.items():
        for candidate in candidates:
            new_total=total+credits*gp(candidate)
            if new_total>770: continue
            new_cost=(cost[0]+abs(candidate-score),cost[1]+(candidate!=score))
            if new_total not in updated or new_cost<updated[new_total][0]:
                updated[new_total]=(new_cost,chosen+[candidate])
    states=updated
scores=states[770][1]
result=original
permitted=set()
for row,score in zip(rows,scores):
    name,hours,credits,degree,old_score,old_points=row.groups()
    old=row.group(0).encode('utf-16-le')
    new=f'{name}\x07{hours}\x07{credits}\x07{degree}\x07{score}\x07{gp(score)/10:.2f}\x07'.encode('utf-16-le')
    assert len(old)==len(new) and result.count(old)==1
    pos=result.index(old)
    permitted.update(pos+i for i,(a,b) in enumerate(zip(old,new)) if a!=b)
    result=result[:pos]+new+result[pos+len(old):]
    if str(score)!=old_score or f'{gp(score)/10:.2f}'!=old_points:
        print(f'{name}: {old_score} / {old_points} -> {score} / {gp(score)/10:.2f}')
    if name in fixed: assert score==fixed[name]
weighted=sum(Decimal(row.group(3))*Decimal(gp(s))/10 for row,s in zip(rows,scores))
credits=sum(Decimal(row.group(3)) for row in rows)
assert weighted==Decimal('77') and credits==Decimal('24')
assert (weighted/credits).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)==Decimal('3.21')
assert result.count('平均绩点：3.21'.encode('utf-16-le'))==1
assert len(result)==len(original)
assert {i for i,(a,b) in enumerate(zip(original,result)) if a!=b}==permitted
target.write_bytes(result)
assert target.read_bytes()==result
assert source.read_bytes()==original
print(f'PASS: weighted points={weighted}, credits={credits}, rounded GPA=3.21; only {len(permitted)} numeric bytes changed.')
print(target.resolve())
