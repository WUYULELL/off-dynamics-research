# server_v15：源状态轨迹可跟踪性筛查

`code/` 是部署到服务器 `$HOME/v15_screen/` 的全部文件，与交付压缩包 `v15_screen.zip`
（SHA-256 `8693555994cf98379ca66fb4c9dbedbb7f38d6d6298a4f6dde1489380f1f07d0`）中的文件逐字节相同，各文件哈希见 `code/FROZEN.json`。

- 代码作者：Claude。通用的一步跟踪程序（Hopper、Walker2d），物理约定、观测反演和求解器只读加载自 `$HOME/v13_action_fix/stage0`，工具函数来自 `$HOME/v14_diag/core.py`。
- 先在 hopper-morph 上与 v14b 的结果做逐轨迹回归检验，一致后才运行 walker2d-morph。
- 状态：尚未在服务器上运行。本地没有 MuJoCo，用合成物理做了端到端测试；其中本包的跟踪程序与 `server_v14/code/diagnostic_b.py` 的逐轨迹结果完全一致。
- 授权范围与预设规则见 `code/INSTRUCTION_V15.md`。
- 归档布局不等于部署布局，不能在本仓库目录内直接运行。`V15_TEST_SHIM` 环境变量只用于本地合成测试，服务器入口会清除它。
