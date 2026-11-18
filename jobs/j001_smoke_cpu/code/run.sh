#!/usr/bin/env bash
# CPU 冒烟测试 + 只读资源盘点。不训练、不写任何已有目录，只写本任务的 outputs/。
set -u
mkdir -p outputs
R="outputs/inventory.txt"
{
  echo "== 基本信息"; date -Is; hostname; echo "HOME=$HOME"; echo "PWD=$PWD"
  echo "CUDA_VISIBLE_DEVICES=[${CUDA_VISIBLE_DEVICES-unset}]"; echo "nproc=$(nproc)"
  echo; echo "== 内存"; free -g
  echo; echo "== 磁盘（HOME）"; df -h "$HOME"
  echo; echo "== GPU 概况（只读查询）"
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv 2>&1
  echo; echo "== 迁移项目（只读）"
  T="$HOME/cluster_zip_reupload_20261104/off-dynamics-transfer-20261104"
  ls -la "$T" 2>&1 | head -40
  echo; echo "-- cluster/paths.example.json"; cat "$T/cluster/paths.example.json" 2>&1 | head -60
  echo; echo "-- MANIFEST.json（前 40 行）"; head -40 "$T/MANIFEST.json" 2>&1
  echo; echo "-- source_snapshot/algo/offline_offline"; ls "$T/source_snapshot/algo/offline_offline" 2>&1 | head -60
  echo; echo "== ODRL 目标数据压缩包"
  ls -la "$HOME/odrl_target_20261115.zip" 2>&1
  echo; echo "== v13 求解与回放代码"; ls "$HOME/v13_action_fix/stage0" 2>&1 | head -40
  echo; echo "== HOME 下的顶层目录（只列名字）"; ls "$HOME" 2>&1 | head -80
} > "$R" 2>&1
python - <<'PY'
import json, os, sys, time
r = {"python": sys.version.split()[0], "executable": sys.executable,
     "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
     "job": os.environ.get("AUTO_JOB_ID"), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
for mod in ["numpy", "scipy", "torch", "gym", "mujoco_py", "d4rl", "h5py"]:
    t = time.time()
    try:
        m = __import__(mod)
        r[mod] = "%s (%.1fs)" % (getattr(m, "__version__", "ok"), time.time() - t)
    except Exception as e:
        r[mod] = "IMPORT FAILED: %s: %s" % (type(e).__name__, str(e)[:200])
try:
    import torch
    r["torch_cuda_built"] = torch.version.cuda
    x = torch.randn(256, 256); r["cpu_matmul_ok"] = bool(torch.isfinite(x @ x).all().item())
except Exception as e:
    r["torch_check"] = "FAILED: %s" % e
json.dump(r, open("outputs/smoke_cpu.json", "w"), indent=2, ensure_ascii=False)
lines = ["# j001_smoke_cpu 冒烟测试报告", ""] + ["- %s: %s" % (k, v) for k, v in r.items()] + ["", "资源盘点见 inventory.txt"]
open("outputs/REPORT_ZH.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
failed = [k for k in ("numpy", "torch", "gym") if str(r.get(k, "")).startswith("IMPORT FAILED")]
sys.exit(1 if failed else 0)
PY
