from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP
from docx import Document
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

OUT = Path('outputs/gpa_reference')
OUT.mkdir(parents=True, exist_ok=True)
courses = [
('高等数理统计',36,2,75),('机器学习',54,3,82),
('科技论文写作（英语）',36,2,86),('学科前沿专题',36,2,89),
('高级算法设计与分析',54,3,66),('深度学习与人工智能',36,2,90),
('云计算技术及应用',36,2,81),('大数据平台技术',36,2,91),
('数字经济与产品思维',36,2,94),('自然辩证法概论',18,1,87),
('新时代中国特色社会主义理论与实践',36,2,79),('工程伦理',18,1,83)]
def gp(s):
    for lower, p in [(90,40),(85,37),(82,33),(78,30),(75,27),(71,23),(66,20),(62,17),(60,13)]:
        if s >= lower: return p
    return 0
# Find a transparent illustration closest to the supplied numeric scores.
states = {0:(0, [])}
for _,_,c,s in courses:
    new = {}
    for total,(cost, vals) in states.items():
        for candidate in range(60,101):
            t = total+c*gp(candidate)
            v = (cost+abs(candidate-s), vals+[candidate])
            if t not in new or v[0] < new[t][0]: new[t] = v
    states = new
cost, scores = states[770]
weighted = sum(c*gp(s) for (_,_,c,_),s in zip(courses,scores))/10
credits = sum(c[2] for c in courses)
assert weighted == 77 and credits == 24
assert (Decimal('77')/Decimal('24')).quantize(Decimal('.01'),rounding=ROUND_HALF_UP)==Decimal('3.21')
doc = Document()
section = doc.sections[0]
section.page_width=Cm(21); section.page_height=Cm(29.7)
section.top_margin=Cm(1.6); section.bottom_margin=Cm(1.6)
section.left_margin=Cm(1.5); section.right_margin=Cm(1.5)
for name in ['Normal','Title','Heading 1']:
    st=doc.styles[name]; st.font.name='Microsoft YaHei'
    st.element.rPr.rFonts.set(qn('w:eastAsia'),'Microsoft YaHei')
    st.font.color.rgb=RGBColor(0,0,0)
doc.styles['Normal'].font.size=Pt(9)
doc.styles['Normal'].paragraph_format.space_after=Pt(6)
doc.styles['Title'].font.size=Pt(19)
doc.add_paragraph('平均绩点 3.21 模拟测算', 'Title')
p=doc.add_paragraph('成绩示例  仅供计算参考  非正式成绩单')
p.runs[0].bold=True
doc.add_paragraph('本表沿用原文件的课程、学时和学分，保留原成绩供对照。模拟成绩和模拟绩点是为演示平均绩点 3.21 而计算的假设值，不代表实际成绩或校方认定结果。')
doc.add_paragraph('测算结果：学分合计 24.0，学分绩点合计 77.00；平均绩点 = 77.00 ÷ 24.0 = 3.208333…，保留两位小数为 3.21。学时仅列示，不重复参与加权。')
table=doc.add_table(rows=1, cols=8)
table.alignment=WD_TABLE_ALIGNMENT.CENTER; table.autofit=False
widths=[.7,6.1,1,1,1.4,1.4,1.2,1.2]
headers=['序号','课程名称','学时','学分','原成绩','模拟成绩','模拟绩点','学分×绩点']
for cell,txt in zip(table.rows[0].cells,headers): cell.text=txt
for i,((name,h,c,original),s) in enumerate(zip(courses,scores),1):
    vals=[str(i),name,str(h),f'{c:.1f}',str(original),str(s),f'{gp(s)/10:.2f}',f'{c*gp(s)/10:.2f}']
    for cell,txt in zip(table.add_row().cells,vals): cell.text=txt
for ri,row in enumerate(table.rows):
    for ci,cell in enumerate(row.cells):
        cell.width=Cm(widths[ci]); cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
        tcPr=cell._tc.get_or_add_tcPr()
        borders=OxmlElement('w:tcBorders')
        for side in ['top','left','bottom','right']:
            el=OxmlElement('w:'+side); el.set(qn('w:val'),'single'); el.set(qn('w:sz'),'4'); el.set(qn('w:color'),'D9D9D9'); borders.append(el)
        tcPr.append(borders)
        if ri==0:
            sh=OxmlElement('w:shd'); sh.set(qn('w:fill'),'E8EEF4'); tcPr.append(sh)
        for p in cell.paragraphs:
            p.paragraph_format.space_before=Pt(5); p.paragraph_format.space_after=Pt(5)
            p.alignment=WD_ALIGN_PARAGRAPH.LEFT if ci==1 else WD_ALIGN_PARAGRAPH.CENTER
            for r in p.runs: r.font.size=Pt(8); r.bold=ri==0
doc.add_paragraph('换算假设与适用范围','Heading 1')
doc.add_paragraph('模拟使用以下百分制参考区间：90–100 → 4.0；85–89 → 3.7；82–84 → 3.3；78–81 → 3.0；75–77 → 2.7；71–74 → 2.3；66–70 → 2.0；62–65 → 1.7；60–61 → 1.3。补考及格的特殊处理不用于本例。')
doc.add_paragraph('贵州大学公开的2018年规定包含两种记载方式：普通教学班按排名比例确定等级；百分制固定区间适用于交换生、联合培养研究生，并用于缓考成绩记载。本例仅借用后一种方式演示数值关系，不能据此认定该生适用此规则。')
doc.add_paragraph('依据：《贵州大学研究生课程成绩记载规定》（贵大研〔2018〕8号）。当前适用规则及课程等级须以研究生院或教务系统确认为准。')
p=doc.add_paragraph('https://gs.gzu.edu.cn/_upload/article/files/08/cf/7f2b598b4dd7ac4f7dfc81687f62/1bc71a8e-2216-45c5-b86b-e5b5bfe48e2a.pdf')
for r in p.runs:r.font.size=Pt(7)
path=OUT/'平均绩点3.21_模拟测算.docx'
doc.save(path)
print(path.resolve())
print('Simulated scores:',scores,'sum:',weighted,'GPA:',weighted/credits)
