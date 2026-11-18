# 自动实验循环：使用说明

```
 Claude（定时会话）           GitHub: auto-loop 分支            实验室电脑（relay，每 1 小时）        服务器
  写 jobs/<任务>/   ──push──▶  jobs/     ──pull──────────────▶  打包上传 → 解压 → tmux 运行   ─────▶  ~/auto_jobs/<任务>/
  读 results/ 分析  ◀──pull──  results/  ◀──push──────────────  取回结果小文件              ◀─────  outputs/、日志
```

- 服务器不连外网，也不连 Claude；实验室电脑只运行一个标准库 Python 脚本，不需要安装 Claude Code。
- Codex 只负责一次性配置（见 `SETUP_CODEX_ZH.md`），日常转发由 `relay/relay.py` 完成。
- 任务格式与授权规则见 `PROTOCOL.md`；Claude 在定时会话里遵守 `CLAUDE_LOOP.md`。

## 你日常需要做的事

1. **在对话里和 Claude 商定实验计划**，写进 `research/PLAN.md`（包括看结果前定好的判定规则）。自动循环只跑登记过的实验。
2. **授权大任务**：收到通知说某任务在 `waiting_approval`，就在 GitHub 网页上编辑 `jobs/<任务名>/job.json`，把 `"approved_by_user": false` 改成 `true`，提交即可。运行上限不超过 24 小时的任务自动运行。
3. **看进展**：`research/LOOP_LOG.md`（每轮结论）、`analysis/`（每个任务的分析）、`results/<任务名>/relay_status.json`（运行状态）。

## 安全边界

- 授权预算、运行时间上限、并发数、GPU 门槛都写在实验室电脑本地的 `relay_config.json` 里，仓库里的任何改动都改不了它们。
- 只有你的 GitHub 账号提交的授权才算数（relay 会核对提交者邮箱）。
- relay 程序本体在实验室电脑上有独立副本，仓库里的 `automation/relay/` 更新后，需要你确认再复制过去才会生效。
- 服务器上每个任务都在新目录 `~/auto_jobs/<任务名>/` 里运行，同名目录已存在时会拒绝运行，不覆盖任何已有文件。
- 只取回小文件（单个 ≤ 5MB，每个任务 ≤ 50MB），不会把数据集或模型推到 GitHub。
- ⚠️ 这个仓库目前是**公开**的，`results/` 里的实验结果任何人都能看到。介意的话，把仓库改回私有。

## 暂停 / 停止

- 暂停全部：实验室电脑上的任务计划程序 → 禁用 `ClaudeRelay`。
- 停掉一个正在跑的任务：在服务器上执行 `bash ~/auto_jobs/_runner/auto_runner.sh kill <任务名>`。
- 停止 Claude 这边：在 claude.ai 的定时任务列表里暂停或删除对应的定时任务。
