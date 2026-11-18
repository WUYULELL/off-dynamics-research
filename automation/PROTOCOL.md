# 任务格式（jobs/ 与 results/ 约定）

## 分工：谁写哪里

| 目录 | 谁写 | 说明 |
|---|---|---|
| `jobs/<任务名>/` | Claude（以及你） | 一个任务一个目录，建好后**不再修改**（授权字段除外） |
| `results/<任务名>/` | 只有中转脚本 relay | 运行状态和取回的结果 |
| `analysis/<任务名>.md` | Claude | 每个任务结束后的分析 |
| `research/PLAN.md` | 你和 Claude（在对话里商定） | 当前预先登记的实验计划与判定规则 |
| `research/LOOP_LOG.md` | Claude | 每轮“假设 → 结果 → 结论 → 下一步” |

各写各的目录，两边同时 push 也不会冲突。

## 一个任务的目录结构

```
jobs/j012_residual_pilot/
├── job.json
└── code/
    ├── run.sh          # 唯一入口；在服务器 ~/auto_jobs/j012_residual_pilot/code/ 下执行
    └── ...             # 本次实验需要的全部新代码
```

- **任务名**：字母、数字、`_ . -`，最长 64 个字符，按 `j编号_简述` 递增。服务器上已有同名目录时会拒绝运行，不会覆盖。
- **不要放数据集和模型**：代码包超过 200MB 会被拒绝。数据和已有代码通过绝对路径读取服务器上现成的文件（只读）。
- `run.sh` 失败时必须以**非 0 退出码**结束，结果写到 `outputs/` 下（默认只取回这个目录）。

## job.json 字段

```json
{
  "job_id": "j012_residual_pilot",
  "description": "这次实验要回答什么问题（一句话）",
  "hypothesis": "预期结果与预先写好的判定规则，或指向 research/PLAN.md 的条目",
  "gpus": 1,
  "max_hours": 6,
  "retries": 1,
  "conda_env": "offdynamics38",
  "result_dirs": ["outputs"],
  "approved_by_user": false,
  "created_by": "claude",
  "parent_job": "j011_xxx"
}
```

| 字段 | 含义 |
|---|---|
| `gpus` | 需要几张卡，0 表示只用 CPU。relay 会选利用率 < 60%、显存占用 < 75% 的卡，并设置 `CUDA_VISIBLE_DEVICES` |
| `max_hours` | 运行时间硬上限，超时会被终止（状态 `timeout`）。也用来计算是否需要你授权 |
| `retries` | 被别人挤掉、服务器重启等**中断**（没有退出码）时自动重跑几次。代码报错（有退出码）不会重跑 |
| `approved_by_user` | **只有你能改**，见下方授权规则 |

## 授权规则（在实验室电脑本地配置，Claude 改不了）

- `max_hours` ≤ `auto_approve_hours`（默认 24 小时）：自动运行。按任务的运行时间上限计算，与用几张卡无关。
- 超过 24 小时：状态变为 `waiting_approval`。你在 GitHub 网页上编辑这个任务的 `job.json`，把 `approved_by_user` 改成 `true` 并提交（随手加一个 `"approved_note": "ok"` 也行）。relay 会检查**最后一次修改 job.json 的提交者邮箱**是否在 `approver_emails` 名单里，所以只有你的提交算数。
- 运行时间上限、并发数、每张卡最多跑几个任务、GPU 门槛，都在实验室电脑的 `relay_config.json` 里，不在仓库里。

## results/<任务名>/relay_status.json 的状态

| 状态 | 含义 |
|---|---|
| `waiting_approval` | 等你授权 |
| `queued` / `waiting_gpu` | 并发已满 / 没有空闲卡，下一轮再试 |
| `running` | 正在服务器 tmux 会话 `auto_<任务名>` 中运行 |
| `completed` | 退出码 0，结果已取回 |
| `failed` | 退出码非 0，或启动失败；`files/_relay/stdout_attempt*.tail.txt` 是日志最后 400 行 |
| `timeout` | 超过 max_hours 被终止，已取回部分结果 |
| `interrupted` | 被中断且重跑次数已用完 |
| `rejected` | 格式错误或服务器上同名目录已存在，没有运行 |

取回的文件在 `results/<任务名>/files/` 下，只取 `.json .md .csv .tsv .txt .yaml .png`，单个文件 ≤ 5MB，单个任务总计 ≤ 50MB。

## 人工干预

- 停掉一个任务：在服务器上执行 `bash ~/auto_jobs/_runner/auto_runner.sh kill <任务名>`，relay 会把它记为 failed（退出码 130，人工停止），不会自动重跑。
- 暂停整个自动化：在任务计划程序里禁用 `ClaudeRelay`。
