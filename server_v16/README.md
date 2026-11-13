# server_v16：不同偏移类型下的源轨迹可跟踪性筛查

`code/` 与交付压缩包 `v16_shift.zip`（SHA-256 `d12693c1eee7b874b2e906b204c65051cb7eacf13e1af55005cd30316062dc0e`）中的文件逐字节相同，`code/assets/` 是 ODRL 仓库的环境 XML 原样副本（MIT 许可）。

- 代码作者：Claude。hopper 与 walker2d 在 ODRL 全部摩擦、重力、运动学、形态偏移仿真器上（含无偏移对照，共 41 个任务），跟踪与 V14b/V15 相同的 500 条 D4RL medium 源轨迹的前 400 步；每个任务带工具自检。不需要目标数据集，不训练。
- 规则在结果产生前写定，见 `code/config.json`。
- 状态：尚未在服务器上运行；本地用合成物理做了端到端测试。
