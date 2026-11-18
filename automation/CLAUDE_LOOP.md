# Claude 定时会话的工作规则（自主实验循环）

每次定时任务唤醒的都是一个**全新会话**，不记得之前的对话。这份文件和 `research/` 下的文件就是全部上下文。

用户的授权（2026-11-18）：**由 Claude 自主设计和推进实验，不需要事先和用户商定计划**；只有运行上限超过 24 小时的任务要等用户授权；达到下方“结束条件”时结束循环。

## 第 0 步：先判断要不要干活（省额度，必须最先做）

1. 拉取 `WUYULELL/off-dynamics-research` 的 `auto-loop` 分支，只读 `research/LOOP_STATE.json`。
2. `loop_enabled` 为 `false`：直接结束，什么都不做，也不通知。
3. 用 `git log -1 --format=%H -- results/` 取最新结果提交，与 `last_seen_results_commit` 比较：
   - **没有变化**，且 `results/_relay_heartbeat.json` 在 12 小时以内：直接结束，不读其他文件，不通知。
   - 心跳超过 12 小时：实验室电脑上的中转可能停了。通知用户一次（`heartbeat_alert_sent` 记为 true，避免重复通知），然后结束。
   - **有变化**：进入第 1 步。

## 第 1 步：读背景

读本文件、`research/CONTEXT.md`、`automation/PROTOCOL.md`、`research/PLAN.md`、`research/LOOP_LOG.md` 最近的条目。按 CONTEXT.md 第 1 节打开“研究进度记录”文档看最新状态；读不到时，在通知里说明，只依据仓库里的文件工作。

## 第 2 步：分析新结果

对每个已结束、但还没有 `analysis/<任务名>.md` 的任务：
- 读 `results/<任务名>/files/` 下的报告、摘要和日志尾部；
- 按 PLAN.md 中**这个任务启动前写好的判定规则**下结论，看到结果后不改规则；
- 写 `analysis/<任务名>.md`：做了什么、数据（含各种子）、按规则的结论、局限；
- 在 `research/LOOP_LOG.md` 末尾追加一条：假设 → 结果 → 结论 → 下一步；
- 在研究进度记录的更新日志里记一行（有工具时）。

## 第 3 步：检查结束条件

对照 `research/PLAN.md` 第一节的“结束条件”。满足任何一条，就把 `loop_enabled` 设为 `false`，写一份总结 `analysis/LOOP_SUMMARY_<日期>.md`，通知用户，然后结束。

## 第 4 步：设计并提交下一个实验（自主）

- **方向范围**：服务于 CONTEXT.md 第 3 节的论文目标和已批准的创新点结构（创新点一、创新点二）。想换到这个范围之外的新方向时，不要自己开始，在通知里向用户提出。
- **先登记，再运行**：每个新实验在提交任务**之前**，先在 PLAN.md 的“实验登记”里追加一条：要回答的问题、任务与设置、种子、预算，以及**判定规则**（写明什么结果算支持、什么算否定）。同一次提交里再放 `jobs/<任务名>/`。
- **筛查和确认分开**：筛查可以用较少的种子快速试；任何“有效”的结论，都必须用**从没用过的新种子**单独做一次确认实验才算数。已用过的种子见 CONTEXT.md（例如 2101–2110、2201–2205 已被 V21 使用），新种子从 3001 开始往上取，并在 LOOP_STATE.json 里记下已用到哪个。
- **对照组**：比较对象至少包括只用目标数据的训练（T）和最强的混合基线（如混合-全部、BC-混合），并在同一台机器上、同一批种子下重跑。
- 代码修复后重跑：新建任务，`parent_job` 写原任务名，description 写清修了什么。
- 新代码放在任务目录里，只读引用服务器上已有的代码和数据，不修改服务器上任何已有文件。
- 固定种子，超参数写进 json；`outputs/summary.json` 由代码直接算好各种子分数、均值、标准差、标准误和回合长度统计。
- 所有种子都保留，不挑种子，不按回报挑模型。
- 本地能测的部分（语法、合成数据上的小测试）先测过再推送。
- `run.sh` 出错时以非 0 退出码结束，并把关键错误打印出来（relay 只取回日志最后 400 行）。
- `gpus` 和 `max_hours` 如实估计。超过 24 小时的任务会停在 `waiting_approval`，此时通知用户。**绝不修改 `approved_by_user` 字段。**
- 同时在排队或运行中的自己的任务，不超过 4 个。

## 第 5 步：收尾

1. 更新 `research/LOOP_STATE.json`：`last_seen_results_commit`、`experiments_started`、`experiments_since_last_positive`、`consecutive_failures`、`next_seed`。
2. 先 `git pull --rebase`，再 `git push`。只改自己负责的内容：`jobs/`（只新增）、`analysis/`、`research/PLAN.md`（只追加）、`research/LOOP_LOG.md`、`research/LOOP_STATE.json`。
3. **通知用户**，只在以下情况通知：有任务需要授权；得到了按规则成立的结论（无论支持还是否定）；循环结束；遇到需要用户决定的问题。内容简短：任务名、一句话结论、关键数字、下一步。只是提交了新实验、没有新结论时不通知。

## 不要做的事

- 不改 `results/`、`automation/relay/`、`automation/server/`。
- 不删除任何任务或结果，不改已经写进 PLAN.md 的登记内容。
- 不为了得到好看的分数改评估协议、判定门槛、基线或种子。
- 不处理 V21（由用户手动运行），除非用户另行登记。
- 不在通知里夸大结果；不确定就写不确定。
