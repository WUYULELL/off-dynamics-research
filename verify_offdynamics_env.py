import importlib
import os
import sys
from pathlib import Path

home = Path.home()
candidate_mujoco = home / ".mujoco210" / "mujoco210"
expected_mujoco = home / ".mujoco" / "mujoco210"

if "MUJOCO_PY_MUJOCO_PATH" not in os.environ and candidate_mujoco.exists():
    os.environ["MUJOCO_PY_MUJOCO_PATH"] = str(candidate_mujoco)

for mujoco_dir in [Path(os.environ.get("MUJOCO_PY_MUJOCO_PATH", "")), expected_mujoco]:
    bin_dir = mujoco_dir / "bin"
    if bin_dir.exists():
        os.environ["PATH"] = str(bin_dir) + os.pathsep + os.environ.get("PATH", "")

packages = [
    "torch",
    "gym",
    "d4rl",
    "mujoco_py",
    "numpy",
    "h5py",
    "tensorboardX",
]

print("Python:", sys.version)
failed = []

for name in packages:
    try:
        module = importlib.import_module(name)
        version = getattr(module, "__version__", "unknown")
        print(f"{name}: OK ({version})")
    except Exception as exc:
        failed.append((name, exc))
        print(f"{name}: FAIL ({exc})")

if failed:
    raise SystemExit(1)

print("ODRL dependency smoke test passed.")
