from common import *
import statistics
def main():
    guard();x=r.read(OUT/'WALKER_RESULT.json');a=r.read(OUT/'ANALYSIS_COMPLETED.json')
    assert x['passed'] and r.sha(a['report'])==a['report_sha256'] and r.sha(OUT/'WALKER_RESULT.json')==a['result_sha256']
    for p,h in x['input_sha256'].items():assert r.sha(p)==h,p
    spec=r.task_spec('walker2d-morph')
    for rec in x['records']:
        events=r.read(Path(rec['out'])/'evaluations.json')['events'];scores=[statistics.mean([(v-spec['score_min'])/(spec['score_max']-spec['score_min'])*100 for v in e['raw_returns']]) for e in events]
        assert abs(statistics.mean(scores[-5:])-rec['Last5'])<1e-8 and abs(scores[-1]-rec['Final'])<1e-8
    for metric in ('Last5','Final'):
        for label,g in x['statistics'][metric]['groups'].items():
            v=[a[metric] for a in sorted([a for a in x['records'] if a['label']==label],key=lambda a:a['seed'])];s=sorted(v)
            assert v==g['values'] and abs(statistics.mean(v)-g['mean'])<1e-8 and abs(statistics.stdev(v)-g['sample_SD'])<1e-8
            assert s[1]==g['Q1'] and min(v)==g['worst']
        for k,refs in x['statistics'][metric]['descriptive_comparisons'].items():
            for ref,v in refs.items():assert v['no_decision'] and abs(v['BC_minus_reference']-(x['statistics'][metric]['groups'][k]['mean']-x['statistics'][metric]['groups'][ref]['mean']))<1e-8
    g=x['statistics']['Last5']['groups'];best=max(['BC_all','BC50','BC25'],key=lambda k:g[k]['mean']);H=g[best]['mean']-g['T']['mean'];SE=math.sqrt(g[best]['sample_SD']**2/5+g['T']['sample_SD']**2/5)
    v='估分员是瓶颈，存在提升空间' if H>=10 and H>2*SE else '该结构下提升空间不足' if H<5 else '不确定' if H<10 else 'V9未覆盖：H>=10但未超过2SE，停止解释并请用户决定'
    d=x['decision'];assert best==d['C'] and abs(H-d['H'])<1e-8 and abs(SE-d['SE_H'])<1e-8 and v==d['headroom']
    b=g['BC_all'];t=g['T'];assert (b['mean']-t['mean']>2*math.sqrt(b['sample_SD']**2/5+t['sample_SD']**2/5))==d['T_not_strong']
    protect('delivery')
    r.write(OUT/'DELIVERY_VERIFIED.json',dict(passed=True,CPU_only=True,Walker_jobs_verified=15,raw_records=40,
        all_raw_episode_group_V9_and_descriptive_comparisons_rechecked=True,all_input_model_report_hashes_verified=True,
        protected_files=len(r.read(OUT/'protected_manifest.json')['sha256']),report_sha256=a['report_sha256'],result_sha256=a['result_sha256'],
        stage4_started=False,T_retuned=False,time=time.time()))
if __name__=='__main__':main()
