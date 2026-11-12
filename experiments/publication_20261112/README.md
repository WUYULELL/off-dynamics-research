# 2026-11-12 追加归档：范围、精确性与隐私

本提交只新增文件，不修改仓库已有文件，不改写历史，不运行实验，不修改服务器或本地的原始科学输出。
现有 experiments/SNAPSHOT_SOURCES.json 保持不动；新目录各有自己的来源清单。

精确副本指原始本地文件或归档 ZIP 成员。经用户要求，绝对路径的账号名替换为 <USER>；Windows 用户路径同样脱敏。每个受影响文件保存原始 source_sha256 与公开 snapshot_sha256、替换类型及次数。源文件没有修改。
用户终端粘贴的报告/JSON 块和从表格直接导出的 CSV 明确标注为提取/派生材料，并保存原附件 SHA-256。原服务器报告哈希能够匹配的另行标记；不能据此声称原始凭据文件字节已经下载。正文中的内部冻结哈希仍指向原始科学文件，而非脱敏副本。

source 为项目相对原路径或归档成员；PROJECT_ROOT 模板为 C:/Users/<USER>/Documents/New project/off-dynamics-rl-main/off-dynamics-rl-main。对终端导出，source_sha256 是重建报告块/所选材料字节的哈希，原始附件哈希另见 attachment_sha256；CSV 另记对应报告哈希。派生表以其导出脚本、主 JSON 及独立长度 JSON 的 SHA-256 追溯。

完整 V11 STAGE1_RESULT.json 超过 5 MB，按用户最新确认未上传，保留在本机原位置。只读导出脚本与小型得分/长度/组别表一并上传；主 JSON 没有的回合长度从独立已验收诊断表关联，缺失组明确标为未记录。
模型、数据集、缓存、逐步损失、安装环境、原始终端登录文本和 ROOT 上游整树不上传。EXCLUDED_FILES.json 列明每个排除源、字节数、SHA-256 与原因。未发现密码、密钥或 token；发现的路径账号名按要求替换，带登录标识的原始粘贴全文不上传。

MISSING_SERVER_ARTIFACTS.json 列出仍在 Linux、未下载的原始材料。server_v12/tools/export_server_small_files.py 可由用户在服务器手动运行，仅读取白名单小文件，过滤敏感信息并输出新 ZIP，不加载模型或调用 GPU。此工具在本次上传期间未运行。
