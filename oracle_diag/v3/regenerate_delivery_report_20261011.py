"""Reissue the v3 delivery report from preserved raw events, without training."""
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
from support_v3 import *
import math, statistics, re, time
from tensorboard.backend.event_processing.event_file_loader import LegacyEventFileLoader


def link(label,path):
    return '['+label+'](<'+Path(path).resolve().as_posix()+'>)'


def main():
    report=HERE/'DELIVERY_REPORT_v3_20261011.md'
    dest=OUT/'delivery_20261011'
    assert not report.exists() and not dest.exists(), 'Preserve previous deliveries'
    review=read(OUT/'final_review2/review.json')
    verified=read(OUT/'final_review2/report_verified.json')
    old_report=HERE/'REPORT_v3.md'
    assert sha(old_report)==verified['report_sha256']
    assert sha(OUT/'final_review2/review.json')==verified['review_sha256']
    assert review['passed'] and read(OUT/'queue_status.json')['state']=='completed'
    formal=read(OUT/'formal_audited.json')
    plan=read(OUT/'recovery2/training_plan.json')
    jobs=[j for j in plan['jobs'] if j['kind']=='formal']
    assert len(jobs)==len(formal['runs'])==15
    means={}; all_rows=[]
    for job,row in zip(jobs,formal['runs']):
        assert (job['method'],job['seed'])==(row['method'],row['seed'])
        done=read(Path(job['out'])/'completed.json')
        assert done['steps']==500000 and done['method']==job['method'] and done['seed']==job['seed']
        assert Path(row['tb_file']).is_relative_to(Path(job['out'])) if hasattr(Path,'is_relative_to') else str(row['tb_file']).startswith(job['out']+os.sep)
        assert sha(row['tb_file'])==row['tb_sha256']
        scores={}
        for event in LegacyEventFileLoader(winpath(row['tb_file'])).Load():
            for value in event.summary.value:
                if value.tag=='test/target_normalized_score':
                    assert event.step not in scores and math.isfinite(value.simple_value)
                    scores[event.step]=float(value.simple_value)
        assert sorted(scores)==list(range(10000,500001,10000))
        last=[scores[s] for s in [460000,470000,480000,490000,500000]]
        avg=statistics.mean(last)
        assert last==row['last5'] and avg==row['last5_mean']
        means[job['method'],job['seed']]=avg
        all_rows.append(dict(method=job['method'],seed=job['seed'],last5=last,last5_mean=avg))
    names=['TARGET','MIX-CONST','MIX-ORACLE-BACKUP']
    short={'TARGET':'TARGET','MIX-CONST':'CONST','MIX-ORACLE-BACKUP':'ORACLE'}
    stats={}
    for m in names:
        x=[means[m,s] for s in range(5)]
        stats[m]=dict(mean=statistics.mean(x),sd=statistics.stdev(x),se=statistics.stdev(x)/math.sqrt(5))
        assert abs(stats[m]['mean']-review['arms'][m]['mean'])<1e-12
    pairs={}
    for a,b in [('MIX-ORACLE-BACKUP','TARGET'),('MIX-CONST','TARGET'),('MIX-ORACLE-BACKUP','MIX-CONST')]:
        k=a+' minus '+b; x=[means[a,s]-means[b,s] for s in range(5)]
        sd=statistics.stdev(x); se=sd/math.sqrt(5); mu=statistics.mean(x)
        p=dict(values=x,mean=mu,sd=sd,se=se,two_se=2*se,positive=sum(t>0 for t in x))
        p['gate']=mu>2*se and p['positive']>=4
        for key in ['mean','se','positive']:
            assert abs(p[key]-review['comparisons'][k][key])<1e-12
        assert p['gate']==review['comparisons'][k]['gate_passed']
        pairs[k]=p
    print('15 raw TensorBoard files, 750 scores and paired statistics reconciled',flush=True)
    text=['''# Oracle v3 实验交付结果报告

**版本：重新整理交付版｜生成日期：2026-10-11（北京时间）**  
实验与完整核验完成日期：2026-10-10。本报告基于同一批已完成实验重新生成；本次重新读取原始 TensorBoard 并复算统计，没有新增训练。

## 一、交付结论

**本次授权任务已完成：阶段 A 验收完成，3 组 GPU smoke 和 15 项正式训练全部完成并通过核验。阶段 C 按约定未执行。**

在 Walker2d-morph 标准迁移方向下，使用模拟器修正 source 的 Q 学习目标，**没有观察到相对 target-only 或直接混数据的稳定增益**：

- 相对 TARGET：ORACLE 平均低 **3.9052 分**，配对标准误 **4.0520**，5 个种子中 3 个为正。
- 相对 CONST：ORACLE 平均低 **0.0247 分**，配对标准误 **3.2894**，5 个种子中 2 个为正。
- 两项均未通过实验开始前确定的门槛。因此，本轮没有支持进入阶段 C 的结果。

**建议：当前暂停扩大 OT/修正器训练，保留这轮完整结果作为路线判断依据。** 这是对当前投入方向的判断，不是证明 source 无用、整个 Bellman 路线无效，或 TARGET 是任何算法的性能上界。

## 二、本轮究竟验证了什么

### 2.1 研究问题

给定大量正常动力学 source 数据和少量 shifted target 数据，如果可以调用 shifted 模拟器，为每条 source 数据提供对应的一步转移，能否让 IQL 在 shifted 环境中获得更好的回报？

这里的 **ORACLE** 指“利用模拟器和恢复的物理状态生成一步标签”，不是完美价值函数。V 仍由训练学习；修正只作用于 source Q 的回归目标。

### 2.2 三组对照

| 方法 | 含义 | 每步 source / target | source Q 标签 |
|---|---|---:|---|
| TARGET | 只使用 target 数据训练 IQL | 0 / 128 | 不使用 source |
| MIX-CONST，简称 CONST | 直接混合两域数据，source 固定系数 0.20 | 64 / 64 | source 原记录 |
| MIX-ORACLE-BACKUP，简称 ORACLE | 相同混合流程，替换 source Q 标签 | 64 / 64 | shifted 模拟器生成 |

MIX-NAIVE 与 MIX-CONST 在本方案中完全相同，因此只保留 CONST 一个对照，没有重复训练第四组。

两混合组的 source Q 标签分别为：

```text
CONST： y_s = r_s + 0.99 × not_done_s × V(next_state_s)
ORACLE：y_s = r°  + 0.99 × not_done°  × V(next_state°)
```

`°` 表示 shifted 模拟器提供的标签。两组使用相同 source 行索引，source actor、V 的更新规则及权重保持一致。损失为 `mean(target_loss) + 0.20 × mean(source_loss)`；0.20 是系数，不是 source 占总损失的归一化比例。

**与原 Bellman/OT 方案的关系：** 本轮是“模拟器提供一步标签是否值得”的诊断，没有运行 kNN 或 OT coupling。它不能直接证明 OT 比 kNN 更好或更差；即使 oracle 组失败，也不能推出所有 Bellman 加权方法均失败。

### 2.3 固定条件

| 项目 | 本轮设定 |
|---|---|
| 主任务与方向 | walker2d-morph；正常 D4RL medium → OTDF shifted medium |
| 正式评估环境 | shifted Walker2d，OTDF walker_morph.xml，TimeLimit=1000 |
| source 数据 | 999322 条保留转移，1191 个轨迹段 |
| target 数据 | 5000 条，约 8 个轨迹段 |
| 算法 | IQL；相同网络结构与优化器参数 |
| 预算与参数 | source 系数 0.20；eta=1；gamma=0.99；总 batch=128 |
| 训练 | seeds 0–4，每组每 seed 500000 步；共 15 项、750 万步 |
| 评估 | 每 10000 步评估 10 个 episode |
| 主指标 | 460k、470k、480k、490k、500k 五次 normalized return 均值（Last-5） |
| loader | explicit_next_v3；target 读取文件显式 next_observations |

source 沿用原 D4RL qlearning_dataset 的保留行口径。source/target 没有交换。主任务归一化为：

`score = (return − 10.079455055289959) / (4398.43 − 10.079455055289959) × 100`

下文的差值都是该分数单位的点差，不是相对百分比。

## 三、阶段完成清单

| 阶段 / 项目 | 状态 | 交付范围 |
|---|---|---|
| A：数据和环境接入 | 完成 | Walker2d/Hopper × morph/gravity 四任务接入与验收资料 |
| A：正常 source 抽样回放 | 完成 | Walker2d、Hopper 各固定 5000 点；历史恢复后全部通过 |
| A：主任务 source oracle | 完成 | 999322 条 source 行的 shifted 一步标签及索引映射 |
| B：GPU smoke | 3/3 完成 | 三组各 seed0、5000 步，仅用于运行验收 |
| B：正式训练 | 15/15 完成 | TARGET、CONST、ORACLE 各五个 seed，各 500k |
| B：最终核验与分析 | 完成 | 原始事件、配对统计、模型、数据对齐及保护哈希 |
| C：无模拟器估计器 | 未执行 | 不在本次执行范围，且 B 未达到正向工程门槛 |

“实验完成”与“方法达到正向门槛”是两个不同结论：本轮实验完整完成，效果门槛未通过。

## 四、正式训练结果

### 4.1 五种子汇总

每个 seed 先计算一个 Last-5 均值，再在五个 seed 间计算平均值、样本标准差（SD）和标准误（SE）。SD 描述不同种子的波动；SE=SD/√5。

| 方法 | Last-5 平均分 | 样本 SD | SE |
|---|---:|---:|---:|''']
    for m in names:
        s=stats[m];text.append('| {} | {:.4f} | {:.4f} | {:.4f} |'.format(short[m],s['mean'],s['sd'],s['se']))
    text+=['','### 4.2 逐 seed 结果和配对差','',
        '| seed | TARGET | CONST | ORACLE | ORACLE−TARGET | CONST−TARGET | ORACLE−CONST |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for seed in range(5):
        t,c,o=[means[m,seed] for m in names]
        text.append('| {} | {:.4f} | {:.4f} | {:.4f} | {:+.4f} | {:+.4f} | {:+.4f} |'.format(seed,t,c,o,o-t,c-t,o-c))
    text+=['','### 4.3 配对统计及判定','',
        '预设工程门槛：**同 seed 配对差的均值 > 2×SE，并且至少 4/5 个 seed 的差值为正。** 使用样本 SD（分母 n−1）。2×SE 是本项目工程门槛，不能称为五种子下严格的 95% 置信界。','',
        '| 比较 | 平均差 | 样本 SD | SE | 2×SE | 正号数 | 判定 |',
        '|---|---:|---:|---:|---:|---:|---|']
    for key,p in pairs.items():
        label=key.replace('MIX-ORACLE-BACKUP','ORACLE').replace('MIX-CONST','CONST').replace(' minus ','−')
        text.append('| {} | {:+.4f} | {:.4f} | {:.4f} | {:.4f} | {}/5 | 未通过，观察均值非正 |'.format(label,p['mean'],p['sd'],p['se'],p['two_se'],p['positive']))
    text+=['''
### 4.4 用容易理解的话解释结果

**ORACLE 与 TARGET：** 三个 seed 有小幅收益，但 seed2、seed3 分别低 18.20 和 7.71 分，最终平均低 3.91 分。这说明收益在这五个 seed 上不稳定；不应只看正向 seed。较低回报的运行通过了数据、日志和模型核验，没有依据把它们删掉。

**ORACLE 与 CONST：** seed0、seed1 分别提高 5.51、8.89 分，seed2、seed3 分别下降 8.56、5.81 分，seed4 接近零。整体差值接近零，但不确定性并不小。因此准确结论是“未观察到稳定附加价值”，不是“已经证明二者等效”。

**CONST 与 TARGET：** 四个 seed 为负，平均低 3.88 分。直接混数据在当前流程下也没有显示收益。这提醒我们：问题未必只在 source Q 的动力学标签，也可能涉及数据覆盖、source actor/V 更新或学得的 V；这些目前仍是候选解释。

## 五、阶段 A 的验收结果

### 5.1 shifted target 回放资料

以下复用先前回放并核对数据/XML 哈希，不把复用记录称为新的回放。验收要求：状态最大绝对误差≤1e−3、奖励误差≤1e−4、终止标记一致。

| 任务 | 通过数量 | 状态最大误差 | 奖励最大误差 | 轨迹段数 |
|---|---:|---:|---:|---:|''']
    for name in ['walker2d-morph','walker2d-gravity','hopper-morph','hopper-gravity']:
        a=read(OUT/'stage_a1'/(name.replace('-','_')+'_acceptance.json'))
        assert a['accepted']
        text.append('| {} | {}/{} | {:.8g} | {:.8g} | {} |'.format(name,a['n'],a['n'],a['state_error']['max'],a['reward_error']['max'],a['trajectories']))
    text+=['''
Walker-gravity 的最大状态误差约 9.34e−4，虽然通过阈值，但不能笼统说所有任务误差都是 1e−6。旧资料没有保存逐行重新观测的 factual done 数组，终止一致性依据原验收记录；这一证据范围保留不变。

### 5.2 正常 source 与主任务 oracle 缓存

- Walker2d、Hopper 各固定抽样 5000 点。首次直接恢复分别有 1560、861 点失败；原因是保存的 qvel 存在截断。保留失败证据，使用轨迹历史恢复后同一批样本全部通过，未修改样本或阈值。
- 主任务完整 source 的 **999322/999322 行**通过正常域事实回放检查；最大状态误差 7.6901e−5，最大奖励误差 6.2627e−6，done 不一致为0。
- 保存同一行对应的 shifted `next_state/reward/not_done`。正常终止513条，shifted终止506条，共17条终止标记发生变化。
- 缓存行映射、有限性和哈希通过；固定 seed=20261008 的256行缓存复算逐位一致。

**验收能说明什么：** 两个原域回放支持环境实现和状态恢复的可信度；同一模拟器的缓存复算支持可重复性。它们不能构成正常 source 在 shifted 环境中反事实转移的独立外部真值验证。

## 六、可信度、运行成本与恢复记录

### 6.1 已完成的完整核验（2026-10-10）

| 核验项 | 结果 |
|---|---|
| 训练和评估完整性 | 15项正式均到500k；750次正式 checkpoint 评估齐全，另有3次smoke评估 |
| 评估对象与归一化 | 运行时确认实际 shifted 对象；XML哈希、lookup key与OTDF常数一致；原始return转分数最大浮点差约4.46e−6 |
| 模型与数值 | 126个最终模型/优化器文件、78个V快照有限；日志simple-value标量无NaN/Inf |
| 初始化、采样与预算 | 同seed初始化相同；混合组首批采样一致；每步域计数及总计数符合128或64+64；source平均权重.20 |
| 数据与oracle对齐 | source保留行、全部字段哈希、oracle索引和缓存哈希一致；target5000行显式loader对齐 |
| 受保护文件 | 1521个旧文件及阶段A冻结的12个v3脚本哈希未变 |

本次重新生成（2026-10-11）核对了上述核验记录与原报告哈希，并重新读取15个原始TensorBoard文件、750个正式分数，复算全部Last-5与五种子配对统计。没有重复训练，也没有把昨天的完整模型/文件核验宣称为今天重新执行。

正式TARGET每项抽取target 6400万次；混合组每项抽取source、target各3200万次。目标数据期望重用次数分别为12800和6400，但反复抽样不能增加独立轨迹数量。只保存了首批索引，不能声称事后逐位核对过全部训练批次。

### 6.2 实际成本

- 15项正式训练进度计时合计 **38.29小时**；首次正式启动到最终完成约 **53.53小时**，后者包含中断及等待。
- 使用RTX 2060 SUPER 8GB，串行训练，每项2个CPU计算线程。进度计时含初始化和过程中评估、不含末次评估尾部，不是纯GPU计算时间。
- 监控快照GPU总占用约1.6GB，不是全过程峰值；PyTorch分配峰值约6–7MB不含CUDA上下文等开销。

### 6.3 中断和修复如何影响结果

1. TensorBoardX把标签空格转为下划线，首版审计未读到分数。只修复审计读取，三组smoke没有重跑。
2. ORACLE seed3在40k时进程退出，原因未明；未发现同期系统重启。原目录 `b1/o3` 保留，同一seed在 `b2/o3` 从0跑满500k。最终统计只用替代完整运行，没有拼接40k，也不是按回报选择重跑。
3. 交付核验的轨迹段计数曾因 `(5000,)` 与 `(5000,1)` 广播失败。新的核验显式展平后仍得到8段；原训练loader本来已展平。失败记录及新attempt都保留，训练协议未变。

## 七、结论的边界与下一步判断

### 7.1 可以成立的结论

- 当前Walker2d-morph、固定数据、.20系数和IQL混合流程下，ORACLE未达到相对TARGET的正向工程门槛。
- 在相同混合流程下，ORACLE相对CONST也未显示稳定附加价值。
- 已完成的结果支持暂不扩大训练、暂不进入阶段C；实验失败门槛不代表实验执行未完成。

### 7.2 不能据此推出的结论

- 不能称TARGET为数学上界，也不能说所有source利用、Bellman加权或OT方案均无效。
- 不能把接近零的ORACLE−CONST点估计当作等效性证明。
- 不能直接与TABB/OTDF论文分数比较优劣：本轮没有完整同协议复现这两个方法。OTDF归一化一致只统一分数单位。

### 7.3 仍未识别的变量

1. target只有约8个轨迹段，5个训练seed没有覆盖独立数据采集的不确定性；Last-5五个checkpoint也不是独立训练。
2. V是学得函数，oracle只提供一步转移，不提供真实最优价值。
3. source actor/V更新未被修正，可能影响负迁移；ORACLE−TARGET还改变了每步target采样数和数据构成。
4. oracle使用恢复的隐藏物理状态和历史信息，不等于只根据单个观测(s,a)可获得的条件转移。

**建议与事实分开：** 本轮事实是没有通过门槛；我的建议是先停止扩大计算投入。如果后续继续，优先针对上述变量做小规模归因，并引入独立target轨迹。具体归因实验尚未执行，不能把任一候选解释当作既定原因。

## 附录 A：Last-5原始分数

每格为该checkpoint的10个episode平均normalized return。统计保留全部预定seed与checkpoint。

| 方法 | seed | 460k | 470k | 480k | 490k | 500k | Last-5 |
|---|---:|---:|---:|---:|---:|---:|---:|''']
    for r in all_rows:
        text.append('| {} | {} | {} | {:.4f} |'.format(short[r['method']],r['seed'],' | '.join('{:.4f}'.format(v) for v in r['last5']),r['last5_mean']))
    text+=['','## 附录 B：证据与文件入口','']
    source_links=[('上一版完整技术报告',old_report),('正式TensorBoard审计与事件文件路径',OUT/'formal_audited.json'),
        ('完整独立核验记录',OUT/'final_review2/review.json'),('报告数值核验记录',OUT/'final_review2/report_verified.json'),
        ('实际训练清单',OUT/'recovery2/training_plan.json'),('冻结协议',OUT/'protocol.json'),
        ('Stage A最终核验',OUT/'stage_a_completed.json'),('全量source oracle缓存记录',OUT/'source_oracle1/completed.json'),
        ('恢复记录',HERE/'STAGE_B_RECOVERY_20261010.md'),('原始用户实验说明',OUT/'USER_SPEC_v3.md')]
    text+=['- '+link(label,path) for label,path in source_links]
    text+=['','**交付状态：** 授权范围内的A、B已完成；C未启动。原始报告、数据、模型、失败证据及恢复产物全部保留。本报告用于本轮结果交付和后续决策，不把局部负结果包装为普适不可能性证明。']
    final='\n'.join(text)+'\n'
    assert '\ufffd' not in final and len(all_rows)==15
    previous=None; tables=0
    for line in final.splitlines():
        if line.startswith('|'):
            if previous is None:tables+=1
            else:assert line.count('|')==previous,line
            previous=line.count('|')
        else:previous=None
    for _,p in source_links: assert p.exists()
    dest.mkdir()
    with report.open('x',encoding='utf-8') as f:f.write(final)
    write(dest/'regeneration_verified.json',dict(passed=True,generated_date='2026-10-11',timezone='Asia/Shanghai',
        report=str(report),report_sha256=sha(report),original_report_sha256=sha(old_report),
        raw_tensorboard_files_verified=15,formal_score_events=750,last5_rows=all_rows,arm_statistics=stats,
        paired_statistics=pairs,tables_checked=tables,all_source_links_exist=True,
        full_audit_reused_from='2026-10-10',training_started=False,stage_c_executed=False,
        script_sha256=sha(__file__),source_receipt_sha256=sha(OUT/'final_review2/review.json'),created=time.time()))
    assert sha(old_report)==verified['report_sha256']
    print(str(report),flush=True)


if __name__=='__main__':main()
