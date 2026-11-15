# server_v17：试点 B（小数据正则化）

`code/` 与交付压缩包 `v17_reg.zip`（SHA-256 `91871951f4ce34ebec45268cbce37338cbd3b3f690a57472eb4fbf5b2a401d6e`）中的文件逐字节相同。

- 代码作者：Claude。在 hopper-morph、walker2d-morph 上只用 5000 条目标数据训练 IQL 和 BC，比较 base、权重衰减、稀疏掩码三种做法，各 5 个种子。
- 网络类和损失函数只读加载自迁移项目中的冻结文件；更新顺序与此前的 T、BC-all 相同。
- 规则在结果产生前写定，见 `code/INSTRUCTION_V17.md` 和 `code/config.json`。
- 状态：尚未在服务器上运行。控制器在本地用替身程序测试过（含外部中断后的重跑）；训练程序因本地没有 PyTorch 未能运行，首次真实运行是服务器上的冒烟测试。
