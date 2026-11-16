# server_v18：试点 A

`code/` 与交付压缩包 `v18_pilotA.zip`（SHA-256 `5160ac02027e3248c362f1f20dd4b2b06b85a7bbc54002bb0a7defd00427f76a`）中的文件逐字节相同；`code/assets/` 是 ODRL 仓库的三个环境 XML 原样副本。目标数据集不在仓库中。

- 代码作者：Claude。在 ODRL 的三个关节限位任务上，比较只用目标数据（T、BC）和混合 D4RL medium 源数据的三种做法，各 5 个新种子。
- 规则在结果产生前写定，见 `code/INSTRUCTION_V18.md`。
- 状态：尚未在服务器上运行。控制器在本地用替身程序测试过；训练程序由已在服务器跑通的 V17 程序改写，带源数据的部分未运行过。
