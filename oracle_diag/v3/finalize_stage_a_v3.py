"""Reconcile immutable Stage A receipts and deliver the Chinese review document."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from support_v3 import *
from task_env import TASKS,paths,call_shifted_env
import numpy as np,h5py,math,time,ast

def link(label,path):return '['+label+'](<'+Path(path).as_posix()+'>)'

def main():
    from stage_a_v3 import step
    from methods_v3 import SourceBackup
    for f in HERE.glob('*.py'):ast.parse(f.read_text(encoding='utf-8'))
    receipt=OUT/'source_oracle1/completed.json';cache=read(receipt)
    assert cache['complete'] and not cache['failed_raw_rows']
    sample=read(OUT/'stage_a2/sample_completed.json');assert sample['all_passed']
    preflight=read(OUT/'preflight1/completed.json');assert preflight['complete'] and preflight['optimizer_steps']==0
    targets={task:read(OUT/'stage_a1'/(task.replace('-','_')+'_acceptance.json')) for task in TASKS}
    scales=read(OUT/'stage_a1/scale.json')['rows']
    ids=np.load(cache['raw_indices_path']);arrays={k:np.load(v['path'],mmap_mode='r') for k,v in cache['arrays'].items()}
    for k,v in arrays.items():
        assert len(v)==999322 and np.isfinite(v).all() and sha(cache['arrays'][k]['path'])==cache['arrays'][k]['sha256']
    assert np.isin(arrays['not_done'],[0,1]).all()
    with h5py.File(sample['sources']['walker2d']['dataset'],'r') as f:
        assert np.array_equal(ids,np.flatnonzero(~f['timeouts'][:-1].astype(bool)))
        check=np.random.default_rng(20261008).choice(len(ids),256,replace=False)
        raw_ids=ids[check];all_actions=f['actions'][:]
    with np.load(str(OUT/'source_oracle1/recovery_evidence.npz')) as z:
        assert not (z['recovery_status'][ids]==2).any()
        assert (z['state_error'][ids]<=1e-3).all() and (z['reward_error'][ids]<=1e-4).all() and not z['done_disagreement'][ids].any()
        qp=z['qpos'][raw_ids];qv=z['qvel'][raw_ids]
    env=call_shifted_env(dict(env_name='walker2d-morph'))
    observed=[step(env,p,v,all_actions[r]) for p,v,r in zip(qp,qv,raw_ids)];env.close()
    assert np.array_equal(np.asarray([r[0] for r in observed],np.float32),arrays['next_state'][check])
    assert np.array_equal(np.asarray([r[1] for r in observed],np.float32).reshape(-1,1),arrays['reward'][check])
    assert np.array_equal((1.-np.asarray([r[2] for r in observed])).reshape(-1,1),arrays['not_done'][check])
    provider=SourceBackup('cpu',receipt);batch=provider(check)
    assert all(np.array_equal(t.numpy(),arrays[k][check]) for t,k in zip(batch,['next_state','reward','not_done']))
    # Recompute stored replay statistics against raw HDF5, independently of report helpers.
    verified=[]
    for kind,s in sample['sources'].items():
        with np.load(s['replay']) as z,h5py.File(s['dataset'],'r') as f:
            raw=z['raw_indices'];truth=f['next_observations'][:][raw];reward=f['rewards'][:][raw];done=f['terminals'][:][raw]
            se=np.max(np.abs(z['replay_next']-truth),axis=1);re=np.abs(z['replay_reward']-reward)
            assert np.array_equal(se,z['state_error']) and np.array_equal(re,z['reward_error']) and np.array_equal(z['replay_done'],done)
            for field,value in [('state_error',se),('reward_error',re)]:
                for name,actual in [('mean',np.mean(value)),('median',np.median(value)),('p95',np.percentile(value,95)),('max',np.max(value))]:
                    assert math.isclose(s[field][name],actual,rel_tol=1e-12,abs_tol=1e-15)
        verified.append(kind)
    for task,t in targets.items():
        assert t['accepted'] and t['n']==5000 and t['buffer_exact_all_rows']
        xml,file=paths(task);assert sha(xml)==t['xml_sha256'] and sha(file)==t['dataset_sha256']
        assert sha(t['previous_replay'])==t['previous_replay_sha256']
        assert sha(t['previous_acceptance'])==t['previous_acceptance_sha256']
    protected_check()
    protected_count=read(OUT/'protected_manifest.json')['count']
    audit=dict(ready_for_training_review=True,oracle_receipt=str(receipt),oracle_receipt_sha256=sha(receipt),
        source_cache_n=len(ids),source_cache_finite_and_aligned=True,source_cache_independent_spot_check_n=256,
        spot_check_seed=20261008,spot_check_bitwise_equal_float32=True,sampled_replay_metrics_recomputed=verified,
        protected_files_unchanged=protected_count,rl_training_steps=0,gpu_training_authorized=False,
        code_sha256={str(f):sha(f) for f in HERE.glob('*.py')},completed=time.time())
    write(OUT/'stage_a_completed.json',audit)
    text=['# Oracle v3：标准方向实验报告','',
        '**阶段 A 已完成，具备提交 GPU 训练审批的条件。阶段 B 尚未启动，阶段 C 按要求不执行。当前没有 v3 策略回报结果。**','',
        '本轮方向：正常 D4RL medium source → OTDF shifted medium target；评估 shifted 环境。所有诊断使用 CPU，所有 JSON/NPZ 产物标记 `explicit_next_v3`。',
        '', '## 1. 方案审查后的修正','',
        '1. v2 的负均值不证明 source 本身没有价值，也不证明 target-only 是任何方法的数学上限。v3 将检验当前协议下模拟器修正是否产生可重复收益。',
        '2. 核对旧实现后，v2 各组总采样量都是 256：TARGET 为 256 target，混合组为 128 source＋128 target。v3 统一总量 128：TARGET 128 target；混合组 64＋64。',
        '3. 文档中的 MIX-NAIVE 与 MIX-CONST 完全相同：全量 source、系数 0.20、记录的转移标签。因此保留一个实验控制，MIX-NAIVE 作为别名，避免重复五个训练。',
        '4. shifted target 的 5000 条原域回放，只能用于验收该模拟器，不能直接拿来给约 100 万条正常 source 提供 oracle 标签。本轮另建完整 source→shifted 缓存并验证行索引。',
        '5. `infos/qvel` 虽以 float64 保存，实际仍在 ±10 截断。首轮将它视为完整状态的假设已被验收否定，原始失败记录保留；随后沿用 v2 的历史恢复流程，没有改样本、阈值或剔除失败行。',
        '', '## 2. 阶段 A：接入与回放验收','',
        '四个任务均通过独立 HDF5→target buffer **全部行**的观测、动作、显式 next_observations、奖励和终止标记对齐检查。严格拒绝未支持的任务名。shifted 环境使用 gym v3 对应类加载 OTDF XML，外包 TimeLimit(1000)。',
        '', '### 2.1 shifted target：复用 v2 验收，不重复回放','',
        '复用前重新核对了原数据和 XML 的 SHA256，并从保存的物理下一状态、奖励重算误差。旧结果没有保存逐行 factual done 数组，所以 done=0 差异的证据来自原验收记录；新 NPZ 明确保存的是 recorded_done，未伪称重新观测。','',
        '| 任务 | 通过/总量 | 状态最大误差 | 奖励最大误差 | 速度截断比例 | 历史修复行数 |',
        '|---|---:|---:|---:|---:|---:|']
    for task,t in targets.items():text.append('| {} | 5000/5000 | {:.6g} | {:.6g} | {:.2%} | {} |'.format(task,t['state_error']['max'],t['reward_error']['max'],t['clipped_fraction'],len(t['repaired_rows'])))
    text+=['','验收阈值保持：状态 L∞≤0.001，奖励绝对误差≤0.0001，终止标记一致。Walker-gravity 最大误差并非 1e-6 量级；不能把文档中“全部 1e-6”作为事实复述。',
        '', '### 2.2 正常 source：同一批 5000 点抽样验收','',
        '| 环境 | 首轮直接恢复失败 | 历史恢复后通过 | 状态最大误差 | 奖励最大误差 | done 差异 |',
        '|---|---:|---:|---:|---:|---:|']
    for kind,s in sample['sources'].items():
        first=read(OUT/'stage_a1'/(kind+'_source_acceptance.json'))
        text.append('| {} | {} | 5000/5000 | {:.6g} | {:.6g} | 0 |'.format(kind,len(first['failed_raw_rows']),s['state_error']['max'],s['reward_error']['max']))
    text+=['','抽样 seed=0；5000 条从保留的 source buffer 行中无放回抽取，两次验收使用相同原始行。历史恢复只沿同轨迹已验证前驱；额外短历史候选长度为 1、2、4、8、16、32、64，观测一致性阈值 1e-4。通过的不同历史还需给出一致的 shifted 转移。',
        '', '### 2.3 主任务完整 source oracle 缓存','',
        '- **{} / {} 条保留 source 行全部通过正常域事实回放检查**；未删样本。'.format(cache['n'],cache['n']),
        '- 状态最大误差 `{:.8g}`；奖励最大误差 `{:.8g}`；done 不一致 `{}`；短历史修复 `{}` 行。'.format(cache['state_error']['max'],cache['reward_error']['max'],cache['done_disagreements'],len(cache['short_history_repairs'])),
        '- oracle 缓存包含全部 source 行对应的 shifted `next_state/reward/not_done`。正常终止 {} 条，shifted oracle 终止 {} 条，其中终止状态变化 {} 条。'.format(cache['normal_terminals'],cache['oracle_terminals'],cache['terminal_changes']),
        '- 保存的 D4RL source 行映射、全部数组有限性、文件哈希均通过；另以 seed=20261008 抽取 256 行重算 shifted 转移，float32 缓存逐位一致。',
        '- source 继续使用原 D4RL qlearning_dataset 口径，target 使用文件显式 next_observations。Walker source 有 513 条 terminal 行的 next_observation 不同，但这些行的原 source bootstrap mask 为 0；非 terminal 行差异为 0。',
        '', '**验收的范围：** 两个原域分别复现记录值，支持环境实现和状态恢复的可信度；同一模拟器生成后再重算的缓存一致性检查也不构成对“source 在 shifted 环境中的真实一步转移”的独立外部验证。观测截断意味着 oracle 还利用了轨迹历史中的额外信息。',
        '', '## 3. 规模与重用次数','',
        '| 任务 | source 条数/轨迹段 | target 条数/轨迹段 | source:target | source 混合重用 | target 混合重用 | target-only 重用 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for r in scales:text.append('| {} | {}/{} | {}/{} | {:.3f}:1 | {:.3f} | {:.0f} | {:.0f} |'.format(r['task'],r['source_n'],r['source_trajectories'],r['target_n'],r['target_trajectories'],r['source_to_target'],r['expected_reuse_source_mixed'],r['expected_reuse_target_mixed'],r['expected_reuse_target_only']))
    text+=['','重用次数=500000×每步该域采样条数/N，是有放回采样下的期望值。source 段数按 terminals/timeouts 划分，target 段数还使用记录的 next_obs 与下一行 obs 不连续边界。**5000 target 行只来自约 7–8 个轨迹段；五个训练 seed 不能代替独立重采数据。**',
        '', '## 4. 阶段 B：已准备，等待确认','',
        '| 组 | 每步 source/target | source V / actor 系数 | source Q 标签 | 正式任务 |',
        '|---|---:|---|---|---|',
        '| TARGET | 0/128 | 无 source | 无 | seeds 0–4，各 500k |',
        '| MIX-CONST（MIX-NAIVE） | 64/64 | 0.20 | 原记录 | seeds 0–4，各 500k |',
        '| MIX-ORACLE-BACKUP | 64/64 | 0.20 | shifted oracle | seeds 0–4，各 500k |',
        '', '仅训练 walker2d-morph，共 15 项正式任务、750 万步；前置三组 seed0 各 5000 步 GPU smoke，smoke 不计入正式成绩。Walker-gravity 和 Hopper 本轮完成接入/验收，不自动扩展正式训练。',
        '', 'source Q 公式：`y_s = r° + 0.99 * not_done° * V(next_state°)`；CONST 用原记录的三元组。混合损失保持 `mean(target_loss) + 0.20 * mean(source_loss)`；0.20 是损失系数，不是归一化后 source 占总损失的严格 20%。actor、V 的 source 标签/权重不随 oracle 改变。',
        '', 'CPU 前向验收已证明：三组网络初始参数一致、CONST/NAIVE 等价、source Q 公式正确、target Q 与 source V/actor 未被替换、采样器与旧版本一致。没有调用 backward 或 optimizer.step。GPU 数值稳定性、训练速度和评估运行时对应关系将在 smoke 中确认。',
        '', '### 4.1 指标与事先确定的判断','',
        '- 主指标：shifted target 的 normalized return，460k、470k、480k、490k、500k 五次评估均值。每次 10 个 episode。实际旧代码 TensorBoard 标签为 `test/target normalized score`（有空格）；审计读取该标签。',
        '- 静态核对原 train.py 的 mode=3 分支：`tar_eval_return = eval_policy(policy, tar_eval_env)` → `get_normalized_score(tar_eval_return, ref_env_name)` → target score 日志。关闭 source/target 交换；运行时还将断言使用的是实际创建的 shifted 对象，并独立核对 raw return→normalized score。',
        '- 主差值：ORACLE − TARGET。五种子配对差均值 > 2×样本SD/√5，且至少 4 个正号，视为通过本轮工程门槛。均值 >0 但不满足门槛标“有限证据”；均值≤0 标“本轮观察均值非正”，不解释为普遍不存在收益。',
        '- 同时报告 CONST − TARGET、ORACLE − CONST。后者才是相同混合流程下动力学修正的增量证据；ORACLE−TARGET 还包含数据构成和 target 每步采样数的变化。',
        '- `2×SE` 是预先约定的工程门槛，不应包装为五种子下严格的 95% 置信区间或普适定理。',
        '', '### 4.2 归一化、时间及显存','',
        '| 任务 | OTDF 最小参考值 | OTDF 最大参考值 |', '|---|---:|---:|']
    for task,t in targets.items():text.append('| {} | {:.12g} | {:.12g} |'.format(task,t['normalization']['min'],t['normalization']['max']))
    text+=['','主任务采用 `(return−10.079455055289959)/(4398.43−10.079455055289959)×100`，不能误用 4592.3。',
        '', '**计划耗时：25–40 小时，仅为排期估算，参考 v2 的 15 项约 38.3 小时；GPU 尚未实测，不能当成保证。** 每次一个 GPU 训练进程、2 个 CPU 计算线程，oracle 数组留在 CPU 的内存映射文件，batch 取出后才搬到 GPU；不构造大 OT 矩阵。smoke 会记录实际显存峰值和时间，并在通过数值、数据、配对与日志审计后才进入正式任务。',
        '', '### 4.3 运行方式','',
        '以下命令仅在本任务得到 GPU/RL 明确确认并记录授权后执行；当前入口会拒绝未授权运行。','',
        '```powershell',"Set-Location -LiteralPath '"+str(ROOT)+"'",
        "& 'D:\\ANACONDA\\envs\\offdynamics\\python.exe' -B oracle_diag/v3/run_queue_v3.py --run",'```',
        '', '队列会保留失败目录，拒绝覆盖非空任务目录，用锁文件防止重复队列。断网不影响已经启动的本地 Python 训练；关机、休眠或进程被结束会中断，因此不要在训练期间重启。',
        '', '## 5. 三个研究问题的当前结论','',
        '1. **oracle 相对 target-only 有多少提升：尚无结果。** 完成 CPU 验收不是回报改善证据，必须等阶段 B 的配对数据。',
        '2. **oracle 相对直接混数据有无附加价值：尚无结果。** MIX-NAIVE=CONST，后续直接报告 ORACLE−CONST。',
        '3. **无模拟器估计器能否接近 oracle：本轮不回答。** 阶段 C 明确不执行；即便未来通过门槛，模拟器真值也只能用于诊断标签，不能泄露给宣称不依赖模拟器的估计器。',
        '', '补充限制：本轮以单任务和五个优化 seed 为主；V 是训练过程中学到的函数；oracle 标签只替换 source Q，source actor/V 仍可能造成负迁移，因此它不是所有方法的数学上界。OTDF 的归一化常数对齐只解决分数单位问题；本实验使用 IQL、固定 .20 系数、64+64 采样、显式 target loader 和模拟器/history oracle，与 OTDF 算法训练协议不同，不能把本轮分数直接与论文表格并列宣称优越。',
        '', '## 6. 可核对的产物','']
    for label,path in [('冻结协议',OUT/'protocol.json'),('状态恢复修正记录',OUT/'stage_a2/protocol_amendment.json'),
        ('source 抽样验收',OUT/'stage_a2/sample_completed.json'),('全量 source oracle',receipt),
        ('规模表',OUT/'stage_a1/scale.json'),('CPU 前向检查',OUT/'preflight1/completed.json'),
        ('具体训练清单',OUT/'training_plan.json'),('最终阶段 A 审计',OUT/'stage_a_completed.json'),
        ('任务适配器',HERE/'task_env.py'),('source Q 接口',HERE/'methods_v3.py'),('训练入口',HERE/'train_entry_v3.py'),
        ('原始用户说明',OUT/'USER_SPEC_v3.md')]:text.append('- '+link(label,path))
    text+=['','四份 shifted target 验收及重放 NPZ 位于 `stage_a1/`；首次正常 source 失败证据也保留在那里。全量 oracle 三个 NPY 数组与恢复状态证据位于 `source_oracle1/`。',
        '', '本轮核对了 {} 个受保护旧文件，SHA256 全部未变。所有新增代码在 oracle_diag/v3，所有新增结果在 oracle_diag/results/v3_20261008。'.format(protected_count),
        '', '## 7. 阶段停点','',
        '依据用户说明：“每个阶段结束时停下来向我汇报，等我确认。没有我的确认，不要启动任何 RL 训练。” 当前停在阶段 A 已验收、阶段 B 待确认的位置；尚未创建 GPU 授权记录。']
    report=HERE/'REPORT_v3.md'
    if report.exists():raise RuntimeError('Refusing report overwrite')
    report.write_text('\n'.join(text)+'\n',encoding='utf-8')
    write(OUT/'status.json',dict(state='stage_a_complete_awaiting_stage_b_confirmation',report=str(report),
        gpu_training_authorized=False,stage_b_started=False,stage_c_executed=False),replace=True)
    print('STAGE A VERIFIED; REPORT',str(report),flush=True)

if __name__=='__main__':main()
