"""Independent-seed confirmation; never triggers estimator development."""
import statistics,math
SEEDS=(201,202,203,204,205)

def paired(a,b,metric='Last5'):
    a=sorted(a,key=lambda x:x['job']['seed']);b=sorted(b,key=lambda x:x['job']['seed'])
    assert [x['job']['seed'] for x in a]==[x['job']['seed'] for x in b]==list(SEEDS)
    d=[x[metric]-y[metric] for x,y in zip(a,b)];m=statistics.mean(d);sd=statistics.stdev(d);se=sd/math.sqrt(5)
    return dict(differences=d,mean=m,sample_SD=sd,SE=se,positive=sum(x>0 for x in d),n=5,
        metric=metric,mean_gt_2SE=m>2*se,positive_at_least4=sum(x>0 for x in d)>=4,
        passed=m>2*se and sum(x>0 for x in d)>=4,
        mean_at_least5=m>=5,strict_95percent_claim=False)

def decide(groups):
    assert set(groups)=={'T','U','G','R'} and all(len(v)==5 for v in groups.values())
    pairs={name:paired(groups[a],groups[b]) for name,a,b in
        [('G_minus_T','G','T'),('G_minus_U','G','U'),('G_minus_R','G','R'),
         ('U_minus_T','U','T'),('R_minus_T','R','T'),('R_minus_U','R','U')]}
    passed=all(pairs[k]['passed'] for k in ('G_minus_T','G_minus_U','G_minus_R'))
    notes=[]
    if passed:
        conclusion='存在小幅稳定的挑选效应'
    elif not pairs['G_minus_R']['passed']:
        conclusion='未建立按覆盖缺口挑选优于同数量随机子集的证据'
    else:
        missing=[]
        if not pairs['G_minus_T']['passed']:missing.append('只用target')
        if not pairs['G_minus_U']['passed']:missing.append('全部保留')
        conclusion='覆盖缺口选择优于同数量随机子集，但未建立相对'+('或相对'.join(missing))+'的稳定收益'
        if not pairs['G_minus_T']['passed']:
            notes.append('G相对R5%的优势可能来自随机小子集本身的损失，例如同更新预算下小子集被高频重复抽样；不能解释为source选择带来正收益。')
        if not pairs['G_minus_U']['passed']:
            notes.append('未建立挑选优于全部保留的证据；不写为两者等效。')
    return dict(status='AWAITING_USER_DECISION',all_three_passed=passed,conclusion=conclusion,
        direction2_ended_by_predefined_decision=not passed,
        next_research_track='user_decision_after_confirmation' if passed else 'direction1_descriptive_evidence',
        paired_Last5=pairs,paired_Final={k:paired(groups[a],groups[b],'Final') for k,a,b in
            [('G_minus_T','G','T'),('G_minus_U','G','U'),('G_minus_R','G','R'),
             ('U_minus_T','U','T'),('R_minus_T','R','T'),('R_minus_U','R','U')]},
        notes=notes,estimator_development_triggered=False,
        original_four_comparisons_possible=False,C_included=False,
        original_minimum5_point_development_threshold_unchanged=True,
        new_small_effect_confirmation_label_is_not_original_development_success=True,
        no_additional_training_in_any_outcome=True,
        future_direction2_restart_requires_new_hypothesis_and_new_data=True,
        no_posthoc_reinterpretation_of_thresholds=True)
