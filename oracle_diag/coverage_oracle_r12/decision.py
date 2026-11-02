"""Fixed validation-only stop rule and prospective interpretation limits."""
import math
import statistics

SEEDS=(101,102,103)
RATIOS=(.05,.2,.5)

def paired(left,right):
    left=sorted(left,key=lambda x:x['job']['seed'])
    right=sorted(right,key=lambda x:x['job']['seed'])
    assert [x['job']['seed'] for x in left]==[x['job']['seed'] for x in right]==list(SEEDS)
    d=[a['Last5']-b['Last5'] for a,b in zip(left,right)]
    sd=statistics.stdev(d)
    return dict(differences=d,mean=statistics.mean(d),sample_SD=sd,SE=sd/math.sqrt(3),
                positive=sum(x>0 for x in d),n=3,descriptive_validation_only=True,
                selected_on_same_validation_seeds=True,strict_95pct_claim=False)

def prospective_interpretation(ratio):
    assert ratio in RATIOS
    return dict(selected_ratio=ratio,
        original_four_formal_rules_still_required=True,
        formal_rules_applied_in_this_stage=False,
        coverage_estimator_eligible_after_future_four_rules=(ratio==.05),
        coverage_estimator_authorized_now=False,
        restriction=('若未来原四项规则在至少两个任务全部通过，按原流程由用户确认后续开发；不据此证明组合效应'
            if ratio==.05 else
            '即使未来四项门槛全部通过也不触发覆盖缺口估计器开发；仅可写访问相关性选择在本协议下有效；先对照OTDF、DVDF相似度筛选重新评估新颖性，由用户另行决定'),
        novelty_review_required_before_further_development=(ratio!=.05),
        novelty_review_methods=['OTDF','DVDF'] if ratio!=.05 else [],
        user_decision_required=True)

def decide(t_runs,u_runs,g_groups):
    assert len(t_runs)==len(u_runs)==3 and sorted(x['ratio'] for x in g_groups)==list(RATIOS)
    for runs in [t_runs,u_runs]+[x['runs'] for x in g_groups]:
        assert sorted(x['job']['seed'] for x in runs)==list(SEEDS)
        assert all(x['job']['steps']==500000 and x['job']['temp']==10 for x in runs)
        assert all(x['last5_steps']==[460000,470000,480000,490000,500000] for x in runs)
    assert all(x['job']['arm']=='T' for x in t_runs)
    assert all(x['job']['arm']=='U' for x in u_runs)
    for group in g_groups:
        assert all(x['job']['arm']=='G' and x['job']['ratio']==group['ratio'] for x in group['runs'])
        assert abs(statistics.mean(x['Last5'] for x in group['runs'])-group['mean'])<1e-10
    best_mean=max(x['mean'] for x in g_groups)
    best=max([x for x in g_groups if abs(x['mean']-best_mean)<=1e-8],key=lambda x:x['ratio'])
    tm=statistics.mean(x['Last5'] for x in t_runs)
    um=statistics.mean(x['Last5'] for x in u_runs)
    passed=best['mean']>tm and best['mean']>um
    diff=paired(best['runs'],t_runs)
    boundary=dict(temp_at_grid_upper_boundary=True,monotonic_500k_grid_means=True,
        strongest_algorithm_family_baseline_not_established=True,no_extra_temperature_search=True,
        future_formal_G_minus_T_interval_closed=[3.,7.],
        primary_reservation_if_future_formal_gap_in_interval=True,
        current_validation_gap_in_interval=3<=diff['mean']<=7,
        T128_descriptive_control_authorized_now=False,
        text='T温度在网格边界10选出，三档均值单调上升，可能未达该算法族最强；未来正式G−T在[3,7]须列为主要保留意见；不追加温度搜索')
    return dict(status='AWAITING_USER_DECISION',passed_validation_stop=passed,
        conclusion='验证阶段见信号' if passed else '验证阶段未见信号',
        stop_rule='selected G mean strictly exceeds both T and U means; equality fails',
        scientific_conclusion=False,formal_gate_applied=False,
        T_mean=tm,U_mean=um,G_best_mean=best['mean'],G_ratio=best['ratio'],G_candidates=g_groups,
        G_minus_T=diff,G_minus_U=paired(best['runs'],u_runs),U_minus_T=paired(u_runs,t_runs),
        prospective_interpretation=prospective_interpretation(best['ratio']),T_boundary_limitation=boundary,
        C_grid_started=False,formal_started=False,task2_started=False,
        estimator_development_started=False,all_stop_outcomes_require_user_decision=True)
