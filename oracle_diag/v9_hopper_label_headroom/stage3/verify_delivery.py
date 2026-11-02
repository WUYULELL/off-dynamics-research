"""Independent CPU arithmetic from raw episode logs; verify report and input hashes."""
from runtime import *
import argparse,statistics
def main(task):
    check_frozen();prefix='HOPPER' if task=='hopper-morph' else 'WALKER'
    result=read(OUT/(prefix+'_RESULT.json'));receipt=read(OUT/(prefix+'_ANALYSIS_COMPLETED.json'))
    assert result['passed'] and receipt['passed'] and receipt['report_sha256']==sha(receipt['report'])
    assert receipt['result_sha256']==sha(OUT/(prefix+'_RESULT.json'))
    for p,h in result['input_sha256'].items():assert sha(p)==h,p
    def recompute(record):
        p=Path(record['out']);events=read(p/'evaluations.json')['events'];spec=task_spec(task)
        values=[statistics.mean([(v-spec['score_min'])/(spec['score_max']-spec['score_min'])*100 for v in e['raw_returns']]) for e in events]
        assert abs(statistics.mean(values[-5:])-record['Last5'])<1e-8 and abs(values[-1]-record['Final'])<1e-8
        assert read(p/'audit.json')['passed'];done=read(p/'completed.json');assert sha(done['final_model'])==done['final_model_sha256']
        return record
    records=[recompute(v) for v in result['records']]
    for metric in ('Last5','Final'):
        for label,group in result['statistics'][metric]['groups'].items():
            v=[r[metric] for r in sorted([r for r in records if r['variant']==label],key=lambda r:r['seed'])]
            assert v==group['values'] and abs(statistics.mean(v)-group['mean'])<1e-8 and abs(statistics.stdev(v)-group['sample_SD'])<1e-8
    g=result['statistics']['Last5']['groups'];best=max(['BC_all','BC50','BC25'],key=lambda k:g[k]['mean'])
    H=g[best]['mean']-g['T']['mean'];SE=math.sqrt(g[best]['sample_SD']**2/g[best]['n']+g['T']['sample_SD']**2/g['T']['n'])
    decision='估分员是瓶颈，存在提升空间' if H>=10 and H>2*SE else '该结构下提升空间不足' if H<5 else '不确定' if 5<=H<10 else 'V9未覆盖：H>=10但未超过2SE，停止解释并请用户决定'
    d=result['decision'];assert best==d['C'] and abs(H-d['H'])<1e-8 and abs(SE-d['SE_H'])<1e-8 and decision==d['headroom']
    B=g['BC_all'];T=g['T'];assert (B['mean']-T['mean']>2*math.sqrt(B['sample_SD']**2/B['n']+T['sample_SD']**2/T['n']))==d['T_not_strong']
    m=read(OUT/'protected_manifest.json')['sha256'];changed=[p for p,h in m.items() if not Path(winpath(p)).exists() or sha(p)!=h];assert not changed
    write(OUT/(prefix+'_DELIVERY_VERIFIED.json'),dict(passed=True,CPU_only=True,raw_records=len(records),independent_raw_episode_arithmetic=True,
        group_and_V9_rules_rechecked=True,all_final_model_and_report_input_hashes_checked=True,protected_files=len(m),changed=changed,
        report_sha256=receipt['report_sha256'],result_sha256=receipt['result_sha256'],time=time.time()))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--task',required=True);main(p.parse_args().task)
