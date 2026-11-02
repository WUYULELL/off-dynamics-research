from common import *
import statistics
def main():
    guard();x=r.read(OUT/'HOPPER_RESULT.json');a=r.read(OUT/'ANALYSIS_COMPLETED.json')
    assert x['passed'] and r.sha(a['report'])==a['report_sha256'] and r.sha(OUT/'HOPPER_RESULT.json')==a['result_sha256']
    for p,h in x['input_sha256'].items():assert r.sha(p)==h,p
    spec=r.task_spec('hopper-morph')
    for record in x['records']:
        events=r.read(Path(record['out'])/'evaluations.json')['events']
        scores=[statistics.mean([(v-spec['score_min'])/(spec['score_max']-spec['score_min'])*100 for v in e['raw_returns']]) for e in events]
        assert abs(statistics.mean(scores[-5:])-record['Last5'])<1e-8 and abs(scores[-1]-record['Final'])<1e-8
    for metric in ['Last5','Final']:
        for label,g in x['statistics'][metric]['groups'].items():
            v=[a[metric] for a in sorted([a for a in x['records'] if a['variant']==label],key=lambda a:a['seed'])]
            assert v==g['values'] and abs(statistics.mean(v)-g['mean'])<1e-8 and abs(statistics.stdev(v)-g['sample_SD'])<1e-8
    g=x['statistics']['Last5']['groups'];best=max(['BC_all','BC50','BC25'],key=lambda k:g[k]['mean']);H=g[best]['mean']-g['T']['mean'];SE=math.sqrt(g[best]['sample_SD']**2/10+g['T']['sample_SD']**2/10)
    assert best==x['decision']['C'] and abs(H-x['decision']['H'])<1e-8 and abs(SE-x['decision']['SE_H'])<1e-8
    verdict='估分员是瓶颈，存在提升空间' if H>=10 and H>2*SE else '该结构下提升空间不足' if H<5 else '不确定' if H<10 else 'V9未覆盖：H>=10但未超过2SE，停止解释并请用户决定'
    assert verdict==x['decision']['headroom']
    for metric in ['Last5','Final']:
        s=x['statistics'][metric]
        for k,by in s['BC_vs_oracle_descriptive'].items():
            for oracle,v in by.items():assert v['no_decision'] and abs(v['BC_minus_oracle']-(s['groups'][k]['mean']-s['groups'][oracle]['mean']))<1e-8
    protected=r.read(OLD/'protected_manifest.json')['sha256'];changed=[p for p,h in protected.items() if not Path(r.winpath(p)).exists() or r.sha(p)!=h];assert not changed
    r.write(OUT/'DELIVERY_VERIFIED.json',dict(passed=True,CPU_only=True,raw_records=60,raw_episode_statistics_and_V9_rechecked=True,
        all_input_model_report_hashes_checked=True,all_six_BC_oracle_comparisons_descriptive_only=True,
        protected_files=len(protected),changed=changed,Walker_runs=0,report_sha256=a['report_sha256'],result_sha256=a['result_sha256'],time=time.time()))
if __name__=='__main__':main()
