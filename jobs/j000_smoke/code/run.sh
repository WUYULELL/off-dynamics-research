#!/usr/bin/env bash
# 冒烟测试：不训练，只检查环境，一分钟内结束。
set -u
mkdir -p outputs
python - <<'PY'
import json, os, platform, socket, sys, time
r = {"host": socket.gethostname(), "python": sys.version.split()[0],
     "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"),
     "job": os.environ.get("AUTO_JOB_ID"), "attempt": os.environ.get("AUTO_ATTEMPT"),
     "cwd": os.getcwd(), "time": time.strftime("%Y-%m-%d %H:%M:%S")}
for mod in ["numpy", "torch", "gym", "mujoco_py", "mujoco", "d4rl"]:
    try:
        m = __import__(mod)
        r[mod] = getattr(m, "__version__", "ok")
    except Exception as e:
        r[mod] = "IMPORT FAILED: %s" % type(e).__name__
try:
    import torch
    r["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        r["gpu_name"] = torch.cuda.get_device_name(0)
        x = torch.randn(1024, 1024, device="cuda"); r["matmul_ok"] = bool((x @ x).isfinite().all().item())
except Exception as e:
    r["torch_check"] = "FAILED: %s" % e
json.dump(r, open("outputs/smoke.json", "w"), indent=2, ensure_ascii=False)
lines = ["# j000_smoke 冒烟测试报告", ""] + ["- %s: %s" % (k, v) for k, v in r.items()]
open("outputs/REPORT_ZH.md", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
PY
echo '{"phase": "COMPLETED"}' > outputs/COMPLETED.json
