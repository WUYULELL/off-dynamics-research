# V12 Linux 部署代码与已收到的结果材料

这是追加的历史快照，不授权启动实验。code/ 为已部署 ZIP 中自有代码、配置和改动版本；historical_code/ 保留早期入口。未上传上游 ROOT 整棵代码树、数据、模型、权重、损失日志或运行中的进程快照。

## 已收到的结果

- 阶段 0：环境回放、GPU 学习器短跑和吞吐测量的整理版文档在 editorial_reports/；不是服务器原始完成凭据。
- 阶段 1A：源数据 BC Last5 13.3235，短回合 100%，整理版文档已保留。
- 阶段 1B：Hopper 2/4/7 轨迹 Last5 为 22.9670 / 36.8060 / 51.0555；Walker 为 24.9557 / 36.4685 / 36.6522。received_reports/stage1B/ 是终端粘贴报告与直接导出的表。
- 阶段 2：ROOT 10k VAE + 20k 主循环是迁移短跑，不是正式结果。整理版文档和部署入口/来源说明已保留。
- 阶段 3：ROOT / T / BC-all，501–505，共 15 项。Last5 为 32.7512 / 41.9092 / 37.6788，H_root 按预设判定不成立。received_reports/stage3/ 报告重建正文的原始 SHA-256 与粘贴的完成凭据一致；公开正文仅按要求替换路径用户名。

terminal export 与服务器原始文件副本明确区分：COMPLETION_FROM_CHAT.json 是用户粘贴的 JSON 块；尚未下载的原始凭据、runtime config / RESULT 不伪造。editorial_reports/ 是此前交付给用户的整理版文档，不冒充原始 REPORT_ZH.md。
完整缺口见 ../experiments/publication_20261112/MISSING_SERVER_ARTIFACTS.json。

历史代码中阶段 0.3/早期阶段 1 的 4/6 并发是当时实际设置；不得将历史配置当作当前服务器使用许可。当前不启动任何实验。后期共享、三进程上限、换卡和中断接续版本均按来源原样归档。

ROOT 上游来源、commit 与有意保持的生效行为见 ROOT_REPRODUCTION_NOTES.md。所有文件来源和哈希见 SNAPSHOT_SOURCES.json。
