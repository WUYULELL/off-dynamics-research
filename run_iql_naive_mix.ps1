# 使用第 0 张 GPU。如果没有 GPU，PyTorch 会自动回退到 CPU，但训练会很慢。
$env:CUDA_VISIBLE_DEVICES = "0"

# ODRL 的 algo/call_algo.py 使用了顶层导入，例如 offline_offline.iql。
# 在 Windows PowerShell 下手动把 .\algo 加到 PYTHONPATH 更稳。
$env:PYTHONPATH = ".\algo;$env:PYTHONPATH"

# IQL + mode 3 = Offline-Offline naive-mix baseline。
python train.py `
  --policy IQL `
  --env walker2d-kinematic-footjnt `
  --shift_level medium `
  --srctype medium `
  --tartype medium `
  --seed 1 `
  --mode 3 `
  --max_step 500000 `
  --dir runs
