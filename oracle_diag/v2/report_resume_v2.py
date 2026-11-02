"""Select the resumed queue and supplemental artifacts for the original report builder."""
from support import *
import report_v2

def main():
    original_get=report_v2.get
    mapping={OUT/'queue_status.json':OUT/'resume1/queue_status.json',
             OUT/'stage_b_new_tasks/completed.json':OUT/'resume1/stage_b_new_tasks/completed.json'}
    def resumed_get(p):return original_get(mapping.get(p,p))
    report_v2.get=resumed_get
    report_v2.main()
    file=HERE/'REPORT_v2.md';text=file.read_text(encoding='utf-8')
    status=read(OUT/'resume1/status.json') if (OUT/'resume1/status.json').exists() else {}
    final=read(OUT/'final_validation.json') if (OUT/'final_validation.json').exists() else {}
    if final.get('passed') and final.get('attempt_directory'):
        status=dict(state='complete (independently verified; original failure retained)')
    notice='\n重启恢复：有效训练队列见 `results/v2_20261006/resume1/queue_status.json`；补充 B 见 `resume1/stage_b_new_tasks/`。原中断目录保留，不计入结果。当前恢复状态：'+status.get('state','preparing')+'。\n'
    text=text.replace('\n## 1.',notice+'\n## 1.',1)
    if final.get('passed') and final.get('attempt_directory'):
        from final_report_notes_v2 import enrich
        text=enrich(text)
    file.write_text(text,encoding='utf-8')
if __name__=='__main__':main()
