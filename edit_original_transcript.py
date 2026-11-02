from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

source = Path(r'C:/Users/TXin/xwechat_files/wxid_3avx27sj4yhi22_2da8/msg/file/2026-09/学生成绩单.doc')
target = Path('outputs/学生成绩单_修改版.doc')
original = source.read_bytes()
assert original[:8] == bytes.fromhex('D0CF11E0A1B11AE1')
updated = original
# Fixed-width substitutions preserve all legacy Word formatting and objects.
records = [
('高等数理统计',36,'2.0','否',75,'2.00',75,'2.70'),
('机器学习',54,'3.0','是',82,'3.30',81,'3.00'),
('科技论文写作（英语）',36,'2.0','是',86,'2.30',86,'3.70'),
('学科前沿专题',36,'2.0','是',89,'3.70',89,'3.70'),
('高级算法设计与分析',54,'3.0','是',66,'1.70',66,'2.00'),
('深度学习与人工智能',36,'2.0','否',90,'3.30',89,'3.70'),
('云计算技术及应用',36,'2.0','否',81,'2.00',81,'3.00'),
('大数据平台技术',36,'2.0','否',91,'4.00',89,'3.70'),
('数字经济与产品思维',36,'2.0','否',94,'3.30',94,'4.00'),
('自然辩证法概论',18,'1.0','否',87,'3.70',87,'3.70'),
('新时代中国特色社会主义理论与实践',36,'2.0','是',79,'2.70',79,'3.00'),
('工程伦理',18,'1.0','否',83,'1.70',83,'3.30'),
]
allowed_positions=set()
def replace_fixed(old,new):
    global updated
    a=old.encode('utf-16-le'); b=new.encode('utf-16-le')
    assert len(a)==len(b)
    assert updated.count(a)==1, (old,updated.count(a))
    pos=updated.index(a)
    for j,(x,y) in enumerate(zip(a,b)):
        if x!=y: allowed_positions.add(pos+j)
    updated=updated[:pos]+b+updated[pos+len(a):]

weighted=Decimal(0)
credits=Decimal(0)
for name,hours,credit,degree,old_score,old_gp,new_score,new_gp in records:
    prefix=f'{name}\x07{hours}\x07{credit}\x07{degree}\x07'
    replace_fixed(prefix+f'{old_score}\x07{old_gp}\x07',prefix+f'{new_score}\x07{new_gp}\x07')
    weighted += Decimal(credit)*Decimal(new_gp)
    credits += Decimal(credit)
    for minimum,points in [(90,'4.00'),(85,'3.70'),(82,'3.30'),(78,'3.00'),(75,'2.70'),(71,'2.30'),(66,'2.00'),(62,'1.70'),(60,'1.30')]:
        if new_score>=minimum:
            assert new_gp==points
            break
gpa=(weighted/credits).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)
assert gpa==Decimal('3.21')
replace_fixed('平均绩点：2.79',f'平均绩点：{gpa}')
assert len(updated)==len(original)
changed={i for i,(a,b) in enumerate(zip(original,updated)) if a!=b}
assert changed==allowed_positions
target.parent.mkdir(exist_ok=True)
target.write_bytes(updated)
assert target.read_bytes()==updated
assert source.read_bytes()==original
print(f'PASS: {len(records)} courses; credits={credits}; weighted points={weighted}; GPA={gpa}')
print(f'PASS: {len(changed)} numeric bytes changed; all other {len(original)-len(changed)} bytes identical; source unchanged.')
print(target.resolve())
