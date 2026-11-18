#!/usr/bin/env python3
"""
relay.py —— 实验室电脑上的中转程序（GitHub <-> 实验服务器）。

只用 Python 标准库，Python >= 3.8，Windows / Linux 都能跑。依赖系统里的 git 和 ssh 命令。
由 Windows 任务计划程序每 10 分钟调用一次：

    python relay.py --config relay_config.json --once

每一轮做的事：
  1. git pull 仓库的自动化分支
  2. 检查服务器上正在跑的任务；结束的把结果小文件取回 results/<任务>/，被挤掉的按规定重跑
  3. 对 jobs/ 下的新任务：检查授权 -> 选空闲 GPU -> 打包上传 -> 服务器解压 -> tmux 启动
  4. 有变化就 commit + push

其他用法：
    python relay.py --config relay_config.json --check     自检（git、ssh、GPU、conda）
    python relay.py --config relay_config.json --loop 600  不用任务计划，前台每 600 秒循环
"""
import argparse
import datetime
import io
import json
import os
import re
import shlex
import subprocess
import sys
import tarfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
JOB_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")
SAFE_REL_RE = re.compile(r"^(?!/)(?!.*(^|/)\.\.(/|$))[A-Za-z0-9_./-]+$")
TEXT_EXT = {".sh", ".py", ".json", ".md", ".yaml", ".yml", ".txt", ".cfg", ".ini", ".csv"}
PENDING = {"queued", "waiting_approval", "waiting_gpu"}
TERMINAL = {"completed", "failed", "timeout", "interrupted", "rejected"}

DEFAULTS = {
    "branch": "auto-loop",
    "remote_root": "auto_jobs",
    "conda_sh": "",
    "default_conda_env": "offdynamics38",
    "ssh": "ssh",
    "git": "git",
    "max_concurrent_jobs": 4,
    "max_jobs_per_gpu": 3,
    "max_gpus_per_job": 4,
    "gpu_util_max": 60,
    "gpu_mem_frac_max": 0.75,
    "auto_approve_gpu_hours": 2.0,
    "hard_max_hours": 48,
    "max_retries_cap": 3,
    "cpu_threads": 4,
    "approver_emails": [],
    "max_file_kb": 5120,
    "max_total_mb": 50,
    "heartbeat_hours": 6,
    "ssh_timeout_sec": 300,
}


def safe_result_name(name):
    if not name or name.startswith("/") or "\\" in name or ":" in name:
        return False
    parts = name.split("/")
    if any(p in ("", ".", "..") for p in parts):
        return False
    return not any(ord(c) < 32 for c in name)


def now_iso():
    return datetime.datetime.now().astimezone().isoformat(timespec="seconds")


class RelayError(Exception):
    pass


class Relay:
    def __init__(self, cfg_path):
        with open(cfg_path, "r", encoding="utf-8") as f:
            user_cfg = json.load(f)
        self.cfg = dict(DEFAULTS)
        self.cfg.update(user_cfg)
        for key in ("repo_dir", "ssh_host"):
            if not self.cfg.get(key):
                raise RelayError("config missing '%s'" % key)
        self.repo = os.path.abspath(os.path.expanduser(self.cfg["repo_dir"]))
        self.jobs_dir = os.path.join(self.repo, "jobs")
        self.results_dir = os.path.join(self.repo, "results")
        self.work_dir = os.path.join(HERE, "relay_work")
        os.makedirs(self.work_dir, exist_ok=True)
        self.log_path = os.path.join(self.work_dir, "relay.log")
        self.runner_local = os.path.join(HERE, "..", "server", "auto_runner.sh")
        self.runner_uploaded = False
        self.changed_paths = set()
        self.commit_notes = []

    # ---------------------------------------------------------------- utils
    def log(self, msg):
        line = "%s %s" % (now_iso(), msg)
        print(line, flush=True)
        try:
            if os.path.exists(self.log_path) and os.path.getsize(self.log_path) > 5 * 1024 * 1024:
                os.replace(self.log_path, self.log_path + ".1")
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def run(self, args, input_bytes=None, timeout=300, check=True):
        try:
            p = subprocess.run(args, input=input_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               timeout=timeout)
        except subprocess.TimeoutExpired:
            raise RelayError("timeout: %s" % " ".join(args[:4]))
        except FileNotFoundError:
            raise RelayError("command not found: %s" % args[0])
        if check and p.returncode != 0:
            raise RelayError("%s failed (rc=%d): %s" % (" ".join(args[:4]), p.returncode,
                                                         p.stderr.decode("utf-8", "replace").strip()[-800:]))
        return p

    def git(self, *args, check=True):
        return self.run([self.cfg["git"], "-C", self.repo] + list(args), check=check, timeout=180)

    def ssh(self, remote_cmd, input_bytes=None, timeout=None, check=True):
        args = [self.cfg["ssh"], "-o", "BatchMode=yes", "-o", "ConnectTimeout=20",
                "-o", "ServerAliveInterval=30", "-o", "ServerAliveCountMax=4",
                self.cfg["ssh_host"], remote_cmd]
        return self.run(args, input_bytes=input_bytes, timeout=timeout or self.cfg["ssh_timeout_sec"],
                        check=check)

    def runner(self, *args, input_bytes=None, timeout=None, check=True):
        self.ensure_runner()
        cmd = "bash ~/%s/_runner/auto_runner.sh %s" % (self.cfg["remote_root"],
                                                      " ".join(shlex.quote(str(a)) for a in args))
        return self.ssh(cmd, input_bytes=input_bytes, timeout=timeout, check=check)

    def ensure_runner(self):
        if self.runner_uploaded:
            return
        with open(self.runner_local, "rb") as f:
            data = f.read().replace(b"\r\n", b"\n")
        root = self.cfg["remote_root"]
        self.ssh("mkdir -p ~/%s/_runner ~/%s/_incoming && cat > ~/%s/_runner/auto_runner.sh.tmp && "
                 "mv -f ~/%s/_runner/auto_runner.sh.tmp ~/%s/_runner/auto_runner.sh"
                 % (root, root, root, root, root), input_bytes=data)
        self.runner_uploaded = True

    # ------------------------------------------------------------------ git
    def pull(self):
        b = self.cfg["branch"]
        dirty = self.git("status", "--porcelain").stdout.decode("utf-8", "replace").splitlines()
        if dirty:
            paths = [line[3:].strip().strip('"') for line in dirty]
            if all(x.startswith("results/") for x in paths):
                # 上一轮中途出错留下的结果文件：先提交，避免丢失
                self.git("add", "-f", "--", "results")
                self.git("commit", "-m", "relay: recover uncommitted results")
                self.log("committed leftover results from an interrupted cycle")
            else:
                raise RelayError("中转用的仓库副本里有非 results/ 的本地改动，为安全起见停止；请人工处理: "
                                 + ", ".join(paths[:5]))
        self.git("fetch", "origin", b)
        cur = self.git("rev-parse", "--abbrev-ref", "HEAD").stdout.decode().strip()
        if cur != b:
            self.git("checkout", b)
        p = self.git("pull", "--rebase", "origin", b, check=False)
        if p.returncode != 0:
            self.git("rebase", "--abort", check=False)
            raise RelayError("git pull --rebase failed: " + p.stderr.decode("utf-8", "replace")[-500:])

    def push(self):
        if not self.changed_paths:
            return
        rels = sorted(os.path.relpath(p, self.repo) for p in self.changed_paths)
        self.git("add", "-f", "--", *rels)
        if self.git("diff", "--cached", "--quiet", check=False).returncode == 0:
            self.changed_paths.clear()
            return
        msg = "relay: " + "; ".join(self.commit_notes[:6])
        if len(self.commit_notes) > 6:
            msg += " (+%d more)" % (len(self.commit_notes) - 6)
        self.git("commit", "-m", msg)
        for attempt in range(3):
            p = self.git("push", "origin", self.cfg["branch"], check=False)
            if p.returncode == 0:
                self.log("pushed: " + msg)
                break
            self.log("push rejected, rebasing (attempt %d)" % (attempt + 1))
            r = self.git("pull", "--rebase", "origin", self.cfg["branch"], check=False)
            if r.returncode != 0:
                self.git("rebase", "--abort", check=False)
                raise RelayError("rebase during push failed")
        else:
            raise RelayError("push failed 3 times")
        self.changed_paths.clear()
        self.commit_notes = []

    # ----------------------------------------------------------------- jobs
    def status_path(self, jid):
        return os.path.join(self.results_dir, jid, "relay_status.json")

    def read_status(self, jid):
        p = self.status_path(jid)
        if not os.path.exists(p):
            return None
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)

    def write_status(self, jid, st, state=None, note="", **fields):
        st = st or {"job_id": jid, "attempts": 0, "history": []}
        old_state, old_note = st.get("state"), st.get("note")
        if state:
            st["state"] = state
        st["note"] = note
        st.update(fields)
        if (state and state != old_state) or note != old_note:
            st.setdefault("history", []).append({"time": now_iso(), "state": st.get("state"), "note": note})
            st["updated_at"] = now_iso()
            p = self.status_path(jid)
            os.makedirs(os.path.dirname(p), exist_ok=True)
            with open(p, "w", encoding="utf-8", newline="\n") as f:
                json.dump(st, f, ensure_ascii=False, indent=2)
                f.write("\n")
            self.changed_paths.add(p)
            self.commit_notes.append("%s -> %s" % (jid, st.get("state")))
            self.log("%s: %s %s" % (jid, st.get("state"), note))
        return st

    def load_jobs(self):
        jobs = {}
        if not os.path.isdir(self.jobs_dir):
            return jobs
        for jid in sorted(os.listdir(self.jobs_dir)):
            d = os.path.join(self.jobs_dir, jid)
            if not os.path.isdir(d) or jid.startswith("_"):
                continue
            jobs[jid] = d
        return jobs

    def validate_job(self, jid, d):
        if not JOB_ID_RE.match(jid):
            return None, "任务名只能包含字母、数字、_ . -，且不超过 64 个字符"
        jp = os.path.join(d, "job.json")
        if not os.path.isfile(jp):
            return None, "缺少 job.json"
        try:
            with open(jp, "r", encoding="utf-8") as f:
                job = json.load(f)
        except ValueError as e:
            return None, "job.json 无法解析: %s" % e
        if not os.path.isfile(os.path.join(d, "code", "run.sh")):
            return None, "缺少 code/run.sh"
        try:
            job["max_hours"] = float(job.get("max_hours", 0))
            job["gpus"] = int(job.get("gpus", 1))
            job["retries"] = int(job.get("retries", 0))
        except (TypeError, ValueError):
            return None, "max_hours / gpus / retries 必须是数字"
        if not (0 < job["max_hours"] <= float(self.cfg["hard_max_hours"])):
            return None, "max_hours 必须在 (0, %s] 之间" % self.cfg["hard_max_hours"]
        if not (0 <= job["gpus"] <= int(self.cfg["max_gpus_per_job"])):
            return None, "gpus 必须在 0..%s 之间" % self.cfg["max_gpus_per_job"]
        job["retries"] = max(0, min(job["retries"], int(self.cfg["max_retries_cap"])))
        dirs = job.get("result_dirs", ["outputs"])
        if not isinstance(dirs, list) or not all(isinstance(x, str) and SAFE_REL_RE.match(x) for x in dirs):
            return None, "result_dirs 必须是代码目录内的相对路径列表"
        job["result_dirs"] = dirs
        env = job.get("conda_env") or self.cfg["default_conda_env"]
        if not re.match(r"^[A-Za-z0-9_.-]+$", env):
            return None, "conda_env 名称不合法"
        job["conda_env"] = env
        return job, ""

    def approval(self, jid, job):
        gpu_hours = job["max_hours"] * max(1, job["gpus"])
        if gpu_hours <= float(self.cfg["auto_approve_gpu_hours"]):
            return True, "自动授权（%.2f GPU·时 ≤ 预算 %.2f）" % (gpu_hours, self.cfg["auto_approve_gpu_hours"])
        if job.get("approved_by_user") is not True:
            return False, "需要你授权：%.2f GPU·时超过自动预算；在 GitHub 上把 job.json 的 approved_by_user 改成 true" % gpu_hours
        approvers = [e.lower() for e in self.cfg.get("approver_emails", []) if e]
        if approvers:
            rel = os.path.relpath(os.path.join(self.jobs_dir, jid, "job.json"), self.repo).replace("\\", "/")
            email = self.git("log", "-1", "--format=%ae", "--", rel).stdout.decode().strip().lower()
            if email not in approvers:
                return False, "approved_by_user=true，但最后修改 job.json 的提交者（%s）不在授权名单里" % email
            return True, "用户授权（%s）" % email
        return True, "用户授权"

    # ------------------------------------------------------------------ gpu
    def pick_gpus(self, n, statuses):
        if n == 0:
            return "none"
        out = self.runner("gpus").stdout.decode().strip().splitlines()
        ours = {}
        for st in statuses.values():
            if st and st.get("state") == "running" and st.get("gpus") not in (None, "none"):
                for g in str(st["gpus"]).split(","):
                    ours[g] = ours.get(g, 0) + 1
        cands = []
        for line in out:
            parts = [x.strip() for x in line.split(",")]
            if len(parts) != 4:
                continue
            try:
                idx, util, used, total = parts[0], float(parts[1]), float(parts[2]), float(parts[3])
            except ValueError:
                continue
            if util >= self.cfg["gpu_util_max"]:
                continue
            if total > 0 and used / total >= self.cfg["gpu_mem_frac_max"]:
                continue
            if ours.get(idx, 0) >= self.cfg["max_jobs_per_gpu"]:
                continue
            cands.append((ours.get(idx, 0), util, used / max(total, 1), idx))
        cands.sort()
        if len(cands) < n:
            return None
        return ",".join(c[3] for c in cands[:n])

    # --------------------------------------------------------------- launch
    def pack(self, jid, d):
        buf = io.BytesIO()
        code_dir = os.path.join(d, "code")
        with tarfile.open(fileobj=buf, mode="w:gz") as tar:
            for root, dirs, files in os.walk(code_dir):
                dirs[:] = sorted(x for x in dirs if x not in ("__pycache__", ".git"))
                for fn in sorted(files):
                    full = os.path.join(root, fn)
                    rel = os.path.relpath(full, code_dir).replace("\\", "/")
                    with open(full, "rb") as f:
                        data = f.read()
                    if os.path.splitext(fn)[1].lower() in TEXT_EXT:
                        data = data.replace(b"\r\n", b"\n")  # Windows 换行会让 bash 脚本报错
                    info = tarfile.TarInfo("code/" + rel)
                    info.size = len(data)
                    info.mode = 0o755 if fn.endswith(".sh") else 0o644
                    info.mtime = int(time.time())
                    tar.addfile(info, io.BytesIO(data))
        return buf.getvalue()

    def launch(self, jid, d, job, st, gpus):
        attempt = int((st or {}).get("attempts", 0)) + 1
        root = self.cfg["remote_root"]
        if attempt == 1:
            data = self.pack(jid, d)
            if len(data) > 200 * 1024 * 1024:
                return self.write_status(jid, st, "rejected", "代码包超过 200MB，请不要把数据或模型放进 jobs/")
            self.ssh("mkdir -p ~/%s/_incoming && cat > ~/%s/_incoming/%s.tar.gz" % (root, root, jid),
                     input_bytes=data, timeout=1800)
            out = self.runner("extract", jid, check=False)
            text = out.stdout.decode().strip()
            if out.returncode == 4 or text == "EXISTS":
                return self.write_status(jid, st, "rejected",
                                         "服务器上 ~/%s/%s 已存在，为防止覆盖拒绝运行；请换一个任务名" % (root, jid))
            if out.returncode != 0:
                raise RelayError("extract failed: " + out.stderr.decode("utf-8", "replace")[-500:])
        maxsec = int(job["max_hours"] * 3600)
        out = self.runner("launch", jid, gpus, maxsec, job["conda_env"], self.cfg["conda_sh"] or "none",
                          attempt, self.cfg["cpu_threads"], check=False)
        text = out.stdout.decode().strip()
        if out.returncode != 0 or text != "LAUNCHED":
            raise RelayError("launch failed: %s %s" % (text, out.stderr.decode("utf-8", "replace")[-500:]))
        return self.write_status(jid, st, "running", "第 %d 次运行，GPU=%s，上限 %g 小时" % (attempt, gpus, job["max_hours"]),
                                 attempts=attempt, gpus=gpus, started_at=now_iso(), max_hours=job["max_hours"],
                                 remote_dir="~/%s/%s" % (root, jid))

    # -------------------------------------------------------------- collect
    def collect(self, jid, job):
        maxkb = int(self.cfg["max_file_kb"])
        p = self.runner("collect", jid, maxkb, *job.get("result_dirs", ["outputs"]), timeout=1800, check=False)
        if p.returncode != 0 and not p.stdout:
            self.log("%s: collect failed: %s" % (jid, p.stderr.decode("utf-8", "replace")[-300:]))
            return "取回结果失败"
        dest = os.path.join(self.results_dir, jid, "files")
        if os.path.isdir(dest):
            for root, dirs, files in os.walk(dest, topdown=False):
                for fn in files:
                    os.remove(os.path.join(root, fn))
                for dn in dirs:
                    os.rmdir(os.path.join(root, dn))
        total, n, skipped = 0, 0, 0
        cap = int(self.cfg["max_total_mb"]) * 1024 * 1024
        try:
            with tarfile.open(fileobj=io.BytesIO(p.stdout), mode="r:gz") as tar:
                for m in tar.getmembers():
                    name = m.name[2:] if m.name.startswith("./") else m.name
                    if not m.isfile() or not safe_result_name(name):
                        skipped += 1
                        continue
                    if total + m.size > cap:
                        skipped += 1
                        continue
                    data = tar.extractfile(m).read()
                    out = os.path.join(dest, *name.split("/"))
                    os.makedirs(os.path.dirname(out), exist_ok=True)
                    with open(out, "wb") as f:
                        f.write(data)
                    self.changed_paths.add(out)
                    total += m.size
                    n += 1
        except (tarfile.TarError, EOFError) as e:
            return "结果包解析失败: %s" % e
        note = "取回 %d 个文件，%.1f MB" % (n, total / 1048576.0)
        if skipped:
            note += "，跳过 %d 个（超出大小上限或路径不安全）" % skipped
        return note

    # ---------------------------------------------------------------- cycle
    def cycle(self):
        self.pull()
        jobs = self.load_jobs()
        statuses = {jid: self.read_status(jid) for jid in jobs}
        valid = {}
        for jid, d in jobs.items():
            job, err = self.validate_job(jid, d)
            st = statuses[jid]
            if job is None:
                if not st or st.get("state") in PENDING:
                    statuses[jid] = self.write_status(jid, st, "rejected", err)
                continue
            valid[jid] = job

        # 1) 检查正在跑的任务
        running = [j for j, st in statuses.items() if st and st.get("state") == "running"]
        if running:
            lines = self.runner("status", *running).stdout.decode().strip().splitlines()
            info = {}
            for line in lines:
                parts = line.split("|")
                if len(parts) == 3:
                    info[parts[0]] = (parts[1] == "1", parts[2])
            for jid in running:
                alive, ec = info.get(jid, (True, ""))
                st = statuses[jid]
                job = valid.get(jid) or {"result_dirs": ["outputs"], "retries": 0}
                if ec == "NODIR":
                    statuses[jid] = self.write_status(jid, st, "failed", "服务器上的任务目录不见了")
                    continue
                if ec == "" and alive:
                    continue
                if ec == "":
                    # tmux 会话没了但没有退出码：被别人结束、服务器重启等
                    if int(st.get("attempts", 1)) <= int(job.get("retries", 0)):
                        st = self.write_status(jid, st, "waiting_gpu", "运行被中断（无退出码），按 retries 设置等待重跑")
                        statuses[jid] = st
                    else:
                        note = self.collect(jid, job)
                        statuses[jid] = self.write_status(jid, st, "interrupted",
                                                          "运行被中断且重跑次数已用完；" + note, finished_at=now_iso())
                    continue
                rc = int(ec)
                note = self.collect(jid, job)
                if rc == 0:
                    state = "completed"
                elif rc == 124:
                    state = "timeout"
                    note = "超过 max_hours 被终止；" + note
                else:
                    state = "failed"
                    note = "退出码 %d；%s" % (rc, note)
                statuses[jid] = self.write_status(jid, st, state, note, exit_code=rc, finished_at=now_iso())

        # 2) 启动新任务
        n_running = sum(1 for st in statuses.values() if st and st.get("state") == "running")
        for jid, job in valid.items():
            st = statuses[jid]
            if st and st.get("state") not in PENDING:
                continue
            ok, why = self.approval(jid, job)
            if not ok:
                statuses[jid] = self.write_status(jid, st, "waiting_approval", why)
                continue
            if n_running >= int(self.cfg["max_concurrent_jobs"]):
                statuses[jid] = self.write_status(jid, st, "queued", "并发已满（%d），排队中" % n_running)
                continue
            gpus = self.pick_gpus(job["gpus"], statuses)
            if gpus is None:
                statuses[jid] = self.write_status(jid, st, "waiting_gpu",
                                                  "没有满足条件的 GPU（利用率<%s%%、显存<%d%%）" %
                                                  (self.cfg["gpu_util_max"], int(self.cfg["gpu_mem_frac_max"] * 100)))
                continue
            try:
                statuses[jid] = self.launch(jid, jobs[jid], job, st, gpus)
                if statuses[jid].get("state") == "running":
                    n_running += 1
                    self.commit_notes.append("approval: " + why)
            except RelayError as e:
                statuses[jid] = self.write_status(jid, st, "failed", "启动失败：%s" % e)

        self.heartbeat(statuses)
        self.push()

    def heartbeat(self, statuses):
        p = os.path.join(self.results_dir, "_relay_heartbeat.json")
        last = None
        if os.path.exists(p):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    last = json.load(f).get("time_epoch")
            except ValueError:
                pass
        if last and time.time() - last < float(self.cfg["heartbeat_hours"]) * 3600 and not self.changed_paths:
            return
        hb = {"time": now_iso(), "time_epoch": int(time.time()),
              "running": sorted(j for j, s in statuses.items() if s and s.get("state") == "running"),
              "pending": sorted(j for j, s in statuses.items() if s and s.get("state") in PENDING)}
        os.makedirs(self.results_dir, exist_ok=True)
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            json.dump(hb, f, ensure_ascii=False, indent=2)
            f.write("\n")
        self.changed_paths.add(p)

    # ---------------------------------------------------------------- check
    def check(self):
        ok = True

        def step(name, fn):
            nonlocal ok
            try:
                r = fn()
                print("[OK]   %s %s" % (name, r or ""))
            except Exception as e:  # noqa
                ok = False
                print("[FAIL] %s: %s" % (name, e))

        step("git 仓库", lambda: self.git("rev-parse", "--show-toplevel").stdout.decode().strip())
        step("git fetch", lambda: self.git("fetch", "origin", self.cfg["branch"]) and self.cfg["branch"])
        step("git push 权限（dry-run）", lambda: self.git("push", "--dry-run", "origin", self.cfg["branch"]) and "ok")
        step("ssh 免密登录", lambda: self.ssh("echo $USER@$(hostname)").stdout.decode().strip())
        step("上传执行器", lambda: self.ensure_runner() or "ok")
        step("tmux", lambda: self.ssh("tmux -V").stdout.decode().strip())
        step("GPU 查询", lambda: " | ".join(self.runner("gpus").stdout.decode().strip().splitlines()))
        if self.cfg["conda_sh"]:
            env = self.cfg["default_conda_env"]
            step("conda 环境 " + env, lambda: self.ssh(
                "source %s && conda activate %s && python -c 'import sys; print(sys.version.split()[0])'"
                % (shlex.quote(self.cfg["conda_sh"]), shlex.quote(env))).stdout.decode().strip())
        else:
            ok = False
            print("[FAIL] conda_sh 未配置")
        print("全部通过" if ok else "有失败项，请修复后重试")
        return ok


def with_lock(work_dir, fn):
    lock = os.path.join(work_dir, "relay.lock")
    if os.path.exists(lock) and time.time() - os.path.getmtime(lock) < 45 * 60:
        print("another relay run is active (lock %s), skip" % lock)
        return
    with open(lock, "w") as f:
        f.write(str(os.getpid()))
    try:
        fn()
    finally:
        try:
            os.remove(lock)
        except OSError:
            pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(HERE, "relay_config.json"))
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--once", action="store_true")
    g.add_argument("--check", action="store_true")
    g.add_argument("--loop", type=int, metavar="SECONDS")
    a = ap.parse_args()
    r = Relay(a.config)
    if a.check:
        sys.exit(0 if r.check() else 1)

    def one():
        try:
            r.cycle()
        except RelayError as e:
            r.log("ERROR " + str(e))
        except Exception as e:  # noqa
            r.log("UNEXPECTED %s: %s" % (type(e).__name__, e))
        finally:
            r.runner_uploaded = False

    if a.loop:
        while True:
            with_lock(r.work_dir, one)
            time.sleep(max(60, a.loop))
    else:
        with_lock(r.work_dir, one)


if __name__ == "__main__":
    main()
