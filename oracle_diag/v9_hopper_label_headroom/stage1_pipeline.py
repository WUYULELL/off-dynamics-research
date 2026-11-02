# -*- coding: utf-8 -*-
"""CPU stage1 cache -> independent replay -> protected delivery -> stop."""
from stage1_runtime import *
import subprocess
import traceback
import numpy as np

def launch(script,name):
    command=[str(PYTHON),'-B',str(HERE/script)]
    with open(wp(OUT/'logs'/(name+'.stdout.log')),'xb') as stdout,open(wp(OUT/'logs'/(name+'.stderr.log')),'xb') as stderr:
        p=subprocess.Popen(command,cwd=str(ROOT),env=os.environ.copy(),stdout=stdout,stderr=stderr,
            creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
        write(OUT/'queue_status.json',dict(stage=name,queue_pid=os.getpid(),active_pid=p.pid,active_command=command,updated=time.time(),RL_training=False),replace=True)
        code=p.wait()
    if code:raise RuntimeError('%s failed with exit%d; no automatic rerun; preserve evidence'%(name,code))

def main():
    check_frozen();start=time.time()
    write(OUT/'queue_started.json',dict(pid=os.getpid(),time=start,command=sys.argv,CPU_only=True))
    launch('stage1_generate.py','cache_generation')
    launch('stage1_accept.py','independent_acceptance')
    check_frozen();g=read(OUT/'CACHE_GENERATED.json');a=read(OUT/'INDEPENDENT_ACCEPTANCE.json')
    assert g['passed'] and a['passed'] and a['sample_n']>=10000
    cfg=read(OUT/'config.json');raw=load_raw();ids=np.load(wp(OUT/'cache/raw_indices.npy'))
    assert np.array_equal(ids,np.flatnonzero(~raw['timeouts'][:-1])) and len(ids)==999998
    for n,r in g['arrays'].items():
        assert sha(r['path'])==r['sha256']
        values=np.load(wp(r['path']),mmap_mode='r');assert len(values)==999998 and np.isfinite(values).all()
    terms=np.load(wp(g['arrays']['terminal']['path']),mmap_mode='r')
    masks=np.load(wp(g['arrays']['not_done']['path']),mmap_mode='r')
    assert np.array_equal(masks,1.-terms) and np.isin(terms,[0,1]).all()
    # Recompute full factual error metrics from saved simulator outputs and raw HDF5.
    actual=np.load(wp(OUT/'cache/recovery/source_replay_next.npy'),mmap_mode='r')
    re_next=np.max(np.abs(actual[ids]-raw['next_observations'][ids]),axis=1)
    actual_reward=np.load(wp(OUT/'cache/recovery/source_replay_reward.npy'),mmap_mode='r')
    re_reward=np.abs(actual_reward[ids]-raw['rewards'][ids])
    for name,value in [('source_replay_state_error',re_next),('source_replay_reward_error',re_reward)]:
        for k,v in stats(value).items():assert np.isclose(g[name][k],v,rtol=0,atol=1e-15)
    with np.load(wp(OUT/'acceptance/independent_replay.npz')) as z:
        ix=z['buffer_indices'];cache_next=np.load(wp(g['arrays']['next_state']['path']),mmap_mode='r')
        cache_reward=np.load(wp(g['arrays']['reward']['path']),mmap_mode='r')
        for name,value in [('target_cache_next_state',np.max(np.abs(z['target_next']-cache_next[ix]),axis=1)),
                           ('target_cache_reward',np.abs(z['target_reward']-cache_reward[ix].ravel()))]:
            assert np.array_equal(value,z[name])
            for k,v in stats(value).items():assert np.isclose(a['errors'][name][k],v,rtol=0,atol=1e-15)
    protected_check('stage_end')
    report=['# V9阶段1：Hopper全量oracle标签与独立回放验收','',
        '**阶段1已完成，通过验收并停止。阶段2/3/4均未启动，0 RL训练、0优化更新，全程CPU。**','',
        '## 范围与方向','',
        '正常D4RL Hopper medium源数据→OTDF Hopper-morph目标动力学。原始1000000行，沿用R14 canonical行999998：仅排除原timeout行及最后无法构成原canonical邻接的一行；未删除任何恢复或回放失败样本。',
        '与旧5000行morph→normal反方向文件无复用关系。源观测/动作不改，oracle只生成原始单位的next_state、reward、terminal与not_done；reward没有新缩放。',
        '目标XML size=.125、timestep=.002、frame_skip4、TimeLimit1000，数据和XML hash均与R14冻结记录匹配。归一化常数取OTDF/info.py，本阶段没有策略评估。','',
        '## 生成与状态恢复','',
        '按Walker已验收做法：未截断速度用原qpos/qvel；截断速度沿同episode、已验证正常域前驱传播；必要时允许原1/2/4/8/16/32/64步历史修复。qpos[0]=0利用平地平移不变性。禁止跨terminal/timeout恢复，禁止放宽误差阈值或去掉失败行。',
        '当前观测恢复阈值1e−4；正常域next-state L∞≤1e−3、奖励绝对误差≤1e−4、terminal逐位一致。target缓存独立回放沿用state≤1e−3、reward≤1e−4、terminal精确一致。所有阈值、样本seed和代码hash在任何新回放前固定。','',
        '|全量canonical生成验收量|均值|中位数|99分位|最大值|','|---|---:|---:|---:|---:|']
    for name,label in [('restored_observation_error','恢复后当前可见观测L∞误差'),('source_replay_state_error','正常源动力学next-state回放L∞误差'),('source_replay_reward_error','正常源动力学reward绝对误差')]:
        r=g[name];report.append('|%s|%.9g|%.9g|%.9g|%.9g|'%(label,r['mean'],r['median'],r['p99'],r['maximum']))
    report+=['',
        'canonical最终恢复失败：0；正常域terminal不一致：%d；短历史额外修复：%d；首次事实门槛未过后修复的行数：%d。'%(g['source_terminal_disagreements'],len(g['short_history_repairs']),len(g['initial_factual_failures'])),
        '恢复状态计数（0直接、1前驱恢复、2失败、3额外短历史修复）：`%s`。'%json.dumps(g['recovery_status_counts']),
        '正常源terminal=%d，morph oracle terminal=%d，二者terminal标签变化%d条。oracle掩码由morph真实回放的terminal重新生成，不继承正常域done。'%(g['source_terminal_count'],g['oracle_terminal_count'],g['terminal_changes']),'',
        '## 独立10000条验收','',
        '样本seed20261030，在999998行中无放回随机抽取10000条，样本索引在生成前封存。独立子进程新建source/target环境，从原始数据寻找同轨迹最近的未截断锚点并重建状态；生成时保存的qpos/qvel从不作为恢复输入，仅在独立回放完成后比较两次恢复的一致性。',
        '|独立验收量（n=10000）|均值|中位数|99分位|最大值|','|---|---:|---:|---:|---:|']
    labels={'restored_observation':'当前可见观测恢复L∞误差','source_next_state':'正常域事实next-state回放误差','source_reward':'正常域事实reward回放误差',
        'target_cache_next_state':'目标morph回放与float32缓存next-state误差','target_cache_reward':'目标morph回放与float32缓存reward误差',
        'restoration_qpos_agreement':'独立恢复与生成恢复qpos差','restoration_qvel_agreement':'独立恢复与生成恢复qvel差'}
    for name,label in labels.items():
        r=a['errors'][name];report.append('|%s|%.9g|%.9g|%.9g|%.9g|'%(label,r['mean'],r['median'],r['p99'],r['maximum']))
    report+=['',
        '独立恢复失败%d条；验收总失败%d条；source terminal不一致%d条；target terminal不一致%d条。'%(a['state_recovery_failed_count'],a['failed_count'],a['source_terminal_disagreements'],a['target_terminal_disagreements']),
        '独立morph回放转为float32后，next-state逐位匹配%d/10000，reward逐位匹配%d/10000。'%(a['bitwise_float32_next_matches'],a['bitwise_float32_reward_matches']),
        '失败条目和原因见INDEPENDENT_ACCEPTANCE.json；没有失败时为空列表。原始回放数组保存在acceptance/independent_replay.npz，可重算上述统计。',
        '状态恢复误差指可见观测的复现，以及正常域事实转移的误差；qpos/qvel差是两次恢复的一致性，不是不可观测速度的外部真值误差。独立回放使用同一MuJoCo目标模型，只验证缓存与恢复的一致性，不构成独立物理世界验证。','',
        '## 耗时、资源与异常','',
        '全量构建%.2f秒（%.2f分钟）；独立验收%.2f秒（%.2f分钟）；本地队列至报告约%.2f秒。阶段开始前仅给出0.5–2小时规划量，本表为实际时间，准备/代码核对及最终人工交付另计。'%(g['elapsed_seconds'],g['elapsed_seconds']/60,a['elapsed_seconds'],a['elapsed_seconds']/60,time.time()-start),
        '显式source/target物理一步调用：生成期`%s`；独立验收`%s`。每个一步调用内部frame_skip4；环境构造各含两个初始化step，单列计入。上述均为模拟器回放，不是训练更新。'%(json.dumps(g['counts']),json.dumps(a['counts'])),
        '本阶段未重跑旧实验、未覆盖任何旧产物。首次事实失败与短历史修复全部留在生成凭据；真实异常由STAGE1_FAILURE.json记录，若存在即不产生成功完成标记。','',
        '## 配置、命令、产物与停点','',
        '代码：`%s`；输出：`%s`。config.json包含全部固定阈值、样本seed、数据方向、dtype与环境定义；code_frozen.json、dependency_hashes.json、PRE_RESULT_COMMITMENT.json固定代码和数据版本。'%(HERE,OUT),
        '执行命令：`D:/ANACONDA/envs/offdynamics/python.exe -B oracle_diag/v9_hopper_label_headroom/stage1_pipeline.py`。CPU子进程由该入口隐藏串行启动，CUDA_VISIBLE_DEVICES=-1；不设置ODRL_PAPER_STYLE_SOURCE_TARGET，无yaml/CLI proportion变更。',
        '缓存cache/raw_indices.npy、next_state.npy、reward.npy、terminal.npy、not_done.npy；生成凭据CACHE_GENERATED.json；独立验收INDEPENDENT_ACCEPTANCE.json；保护凭据protected_stage_end.json；最终STAGE1_COMPLETED.json。',
        '本阶段没有策略训练seed，Last5/Final、训练seed SD及策略分数配对差/SE/正号均不适用。没有作标签效应A/B或BC上限判定。',
        '**按v9阶段1结束后停止，等待阶段2的明确确认。** 后续若启动标签归因，O1与O.25各10seed、500k/temp10，必须逐项与R14核对学习器、64target+64source、source仅V/Q、actor仅target以及target.mean+w*source.mean损失；本阶段未派发这些训练。',
        '阶段2原规划约48小时；可先O1十项、约24小时，完成后再由用户决定O.25。GPU训练的新耗时需启动前核对，缓存构建完成不自动授权训练。','']
    report_path=OUT/'STAGE1_REPORT_ZH.md'
    with open(wp(report_path),'x',encoding='utf-8') as f:f.write('\n'.join(report))
    frozen_code=read(OUT/'code_frozen.json')
    arrays={k:r for k,r in g['arrays'].items()}
    write(OUT/'STAGE1_COMPLETED.json',dict(passed=True,status='AWAITING_STAGE2_CONFIRMATION',CPU_only=True,canonical_rows=999998,
        independent_sample_n=10000,independent_acceptance_passed=True,report=str(report_path),report_sha256=sha(report_path),
        config_sha256=sha(OUT/'config.json'),generated_receipt_sha256=sha(OUT/'CACHE_GENERATED.json'),
        acceptance_sha256=sha(OUT/'INDEPENDENT_ACCEPTANCE.json'),cache_arrays=arrays,
        code_frozen_sha256=sha(OUT/'code_frozen.json'),protected_count=read(OUT/'protected_stage_end.json')['count'],
        training_updates=0,stage2_started=False,stage3_started=False,stage4_started=False,time=time.time()))
    write(OUT/'queue_status.json',dict(stage='AWAITING_STAGE2_CONFIRMATION',queue_pid=os.getpid(),updated=time.time(),report=str(report_path),RL_training=False),replace=True)
    print('STAGE1 COMPLETED; STOP; '+str(report_path),flush=True)

if __name__=='__main__':
    try:main()
    except BaseException:
        path=OUT/('STAGE1_FAILURE_%d.json'%int(time.time()))
        write(path,dict(pid=os.getpid(),time=time.time(),traceback=traceback.format_exc(),action='Preserve evidence, CPU diagnosis; no automatic re-execution or training'))
        write(OUT/'queue_status.json',dict(stage='FAILED',queue_pid=os.getpid(),updated=time.time(),failure=str(path),RL_training=False),replace=True)
        raise
