# 给 Codex 的配置任务：在实验室 Windows 电脑上部署自动中转（relay）

> 把这整份文件发给 Codex。它需要在这台电脑上执行命令；凡是需要输入服务器密码的地方，**由用户本人在终端里输入**，不要把密码写进任何文件、命令行参数或对话。

## 目标

让这台电脑每 1 小时自动执行一次 `relay.py`：从 GitHub 仓库 `WUYULELL/off-dynamics-research` 的 `auto-loop` 分支拉取新实验，上传到实验服务器解压并在 tmux 里运行，跑完后把结果小文件取回并推送到 GitHub。relay 的代码已经写好并测试过，**你的任务只是配置环境，不要修改 `relay.py` 和 `auto_runner.sh` 的逻辑**。如果发现必须改，停下来告诉用户原因。

## 目录约定（路径里不要有中文和空格）

```
D:\claude_relay\
├── off-dynamics-research\    ← 专门给 relay 用的仓库副本（auto-loop 分支），不要在里面手动改东西
└── relay\                    ← relay 程序本体，从仓库复制出来，和仓库副本分开
    ├── relay\relay.py, relay_config.json, run_relay.bat
    └── server\auto_runner.sh
```

**为什么要分开**：relay 程序如果放在仓库副本里，任何推送到仓库的改动都会在下一次 pull 时改掉 relay 自己的代码，包括授权逻辑。分开后，只有用户主动复制才会更新 relay 程序。

盘符不是 D 就换成实际可用的盘。

## 步骤

### 1. 检查前置条件

```powershell
git --version
ssh -V
python --version     # 需要 3.8 及以上；若有多个 Python，记下要用的 python.exe 完整路径
```

- 如果没有 `ssh`：设置 → 应用 → 可选功能 → 添加 “OpenSSH 客户端”。也可以用 Git for Windows 自带的 `C:\Program Files\Git\usr\bin\ssh.exe`（那样要在 relay_config.json 里把 `"ssh"` 设成这个完整路径）。
- relay 只用 Python 标准库，不需要安装任何包。

### 2. 配置 SSH 免密登录

服务器地址、端口、用户名：在 FinalShell 里打开这台服务器的连接设置就能看到，向用户确认。

```powershell
ssh-keygen -t ed25519 -f $env:USERPROFILE\.ssh\labserver_ed25519 -N '""' -C "lab-relay"
```

把公钥装到服务器上（**这一步会提示输入服务器密码，由用户本人输入**）：

```powershell
type $env:USERPROFILE\.ssh\labserver_ed25519.pub | ssh -p <端口> <用户名>@<服务器IP> "mkdir -p ~/.ssh && chmod 700 ~/.ssh && cat >> ~/.ssh/authorized_keys && chmod 600 ~/.ssh/authorized_keys"
```

在 `%USERPROFILE%\.ssh\config` 中追加（文件不存在就新建，用 UTF-8 无 BOM 编码保存）：

```
Host labserver
    HostName <服务器IP>
    Port <端口>
    User <用户名>
    IdentityFile ~/.ssh/labserver_ed25519
    IdentitiesOnly yes
    ServerAliveInterval 30
    ServerAliveCountMax 4
```

验证（必须不提示密码，直接输出用户名和主机名）：

```powershell
ssh -o BatchMode=yes labserver "echo $USER@$(hostname); tmux -V; nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total --format=csv,noheader"
```

- 仍然要求密码：服务器可能禁止公钥登录。**不要**改用保存密码的方案，停下来告诉用户。
- `tmux` 找不到：告诉用户，relay 依赖 tmux。

### 3. 找到服务器上的 conda.sh 和环境名

```powershell
ssh labserver "ls ~/miniconda3/etc/profile.d/conda.sh ~/anaconda3/etc/profile.d/conda.sh /opt/conda/etc/profile.d/conda.sh 2>/dev/null; bash -ic 'conda info --base; conda env list' 2>/dev/null"
```

记下 `conda.sh` 的绝对路径（如 `/home/<用户名>/miniconda3/etc/profile.d/conda.sh`），并确认实验环境名（应为 `offdynamics38`）。以你之前写的服务器交接文档为准。

### 4. 准备仓库副本

```powershell
mkdir D:\claude_relay
cd D:\claude_relay
git clone -b auto-loop https://github.com/WUYULELL/off-dynamics-research.git
cd off-dynamics-research
git config core.autocrlf false
git config user.name "lab-relay"
git config user.email "lab-relay@users.noreply.github.com"
git config pull.rebase true
```

推送权限：让用户在 GitHub 上创建一个 **Fine-grained personal access token**，只授权 `off-dynamics-research` 这一个仓库，权限 `Contents: Read and write`。第一次 push 时 Git Credential Manager 会弹窗，由用户登录或粘贴 token。验证：

```powershell
git push --dry-run origin auto-loop
```

### 5. 复制 relay 程序并填写配置

```powershell
mkdir D:\claude_relay\relay
xcopy /E /I D:\claude_relay\off-dynamics-research\automation\relay D:\claude_relay\relay\relay
xcopy /E /I D:\claude_relay\off-dynamics-research\automation\server D:\claude_relay\relay\server
cd D:\claude_relay\relay\relay
copy relay_config.example.json relay_config.json
```

编辑 `relay_config.json`：

- `repo_dir`：`D:/claude_relay/off-dynamics-research`（用正斜杠）
- `ssh_host`：`labserver`
- `conda_sh`：第 3 步找到的绝对路径
- `default_conda_env`：`offdynamics38`
- `approver_emails`：用户在 GitHub 网页上提交时使用的邮箱。确认方法：请用户在 GitHub 网页上随便编辑一下 `research/LOOP_LOG.md` 并提交，然后执行
  `git -C D:\claude_relay\off-dynamics-research pull` 和 `git -C D:\claude_relay\off-dynamics-research log -1 --format=%ae`，把输出的邮箱填进去（通常形如 `数字+用户名@users.noreply.github.com`）。
- `exclude_gpus`：用户手动运行的实验正在占用的卡号，例如 `["0"]`，自动任务不会分配到这些卡上；没有就留空 `[]`。向用户确认。
- 其余数值保持默认，除非用户另有要求。

编辑 `run_relay.bat`，把 `set PYTHON=python` 改成 python.exe 的完整路径。

### 6. 自检

```powershell
cd D:\claude_relay\relay\relay
python relay.py --config relay_config.json --check
```

必须看到每一项都是 `[OK]`，最后一行是“全部通过”。有失败项就修复，修不好就把输出交给用户。

### 7. 手动跑一轮，验证冒烟测试

仓库里已经有一个冒烟测试任务 `jobs/j000_smoke`（不训练，1 分钟内结束，自动授权）。

```powershell
python relay.py --config relay_config.json --once
# 等 1～2 分钟
python relay.py --config relay_config.json --once
```

第二次应该看到 `j000_smoke: completed`，GitHub 上出现 `results/j000_smoke/files/code/outputs/REPORT_ZH.md`。把这份报告的内容给用户看。

### 8. 注册定时任务

```powershell
schtasks /Create /TN "ClaudeRelay" /SC HOURLY /MO 1 /TR "\"D:\claude_relay\relay\relay\run_relay.bat\"" /F
schtasks /Run /TN "ClaudeRelay"
```

- 默认只在用户登录时运行，所以这台电脑要保持开机并登录。
- 让用户确认电脑不会自动睡眠（设置 → 系统 → 电源；改之前先确认实验室允许）。
- 日志：`D:\claude_relay\relay\relay\relay_work\relay.log`。

### 9. 交付给用户

告诉用户：
1. 自检结果和冒烟测试报告；
2. relay 程序、仓库副本、日志的位置；
3. 如何暂停：任务计划程序 → 禁用 `ClaudeRelay`；
4. 以后如果仓库里的 `automation/relay/` 有更新，需要用户确认后再重新复制第 5 步的两个目录（`relay_config.json` 不要覆盖）。

## 禁止事项

- 不要把服务器密码或 GitHub token 写进任何文件或脚本。
- 不要修改服务器上已有的项目目录、conda 环境和其他用户的进程。
- 不要在 `D:\claude_relay\off-dynamics-research` 里手动改文件或提交（只有 relay 用它）。
- 不要修改 `relay.py` / `auto_runner.sh` 的逻辑；需要改时先停下来告诉用户。
