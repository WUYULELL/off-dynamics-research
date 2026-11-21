# j001_smoke_cpu 冒烟测试报告

- python: 3.8.13
- executable: ~/conda-envs/offdynamics38/bin/python
- CUDA_VISIBLE_DEVICES: 
- job: j001_smoke_cpu
- time: 2026-11-18 23:11:31
- numpy: 1.23.5 (0.1s)
- scipy: 1.4.1 (0.8s)
- torch: 1.11.0+cu113 (0.6s)
- gym: 0.18.3 (0.1s)
- mujoco_py: IMPORT FAILED: Exception: 
You appear to be missing MuJoCo.  We expected to find the file here: ~/.mujoco/mujoco210

This package only provides python bindings, the library must be installed se
- d4rl: IMPORT FAILED: ModuleNotFoundError: No module named 'd4rl'
- h5py: 3.11.0 (0.0s)
- torch_cuda_built: 11.3
- cpu_matmul_ok: True

资源盘点见 inventory.txt
