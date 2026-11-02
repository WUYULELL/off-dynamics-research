# Off-dynamics reinforcement learning research

This private repository tracks the current research code, configurations, accepted protocols and small result summaries built on [OffDynamicsRL/ODRL](https://github.com/OffDynamicsRL/off-dynamics-rl). The upstream README and MIT LICENSE are preserved unchanged. Third-party source notices remain alongside their code.

## Current checkpoint — 2026-11-02

- V9 Hopper and Walker BC checks are complete.
- V11 revised Stage0 is complete, audited and independently verified. No Stage1 or Stage2 has started or been authorized.
- The accepted actor loss is the original O1 advantage-weighted action MSE, `exp(10*A).clamp(max=100)`. SA-top averages over selected samples, returns zero when no source sample is selected, and reports the zero-selection batch fraction.
- Current Stage0 implementation: `oracle_diag/v11_source_actor_recovery_20261102_stage0/`.
- Accepted protocol/configuration/report copies: `experiments/v11/stage0_20261102/`.
- Later experiment order remains a user decision. Historical scripts preserve their historical authorization state; their presence here does not authorize rerunning them.

## Layout

| Path | Contents |
| --- | --- |
| `algo/`, `config/`, `envs/` | Upstream and modified algorithms, configuration and target environment definitions |
| `oracle_diag/` | Research preparation, runners, analysis, audit and verification source; large `results/` excluded |
| `reproductions/OTDF/`, `reproductions/otdf_setup/` | Referenced OTDF source/XML and local setup scripts; environments and datasets excluded |
| `src/` | Referenced third-party Python/source/XML and existing notices; binary assets excluded |
| `experiments/` | Read-only copies of small protocols, configuration, reports and verification receipts |

## Reproducibility and local data

This is a source and experiment-record repository, not a backup of trained models or datasets. HDF5 data, accepted oracle arrays, model checkpoints, TensorBoard files, per-step raw diagnostics and Python environments remain in their original local directories. No original experiment output is moved, deleted or rewritten by this upload.

Frozen runners intentionally validate exact file hashes and some original Windows paths. Cloning the source alone does not restore a runnable frozen experiment: external data/cache files and the accepted directory layout are also required. Do not bypass guards or change frozen scripts just to make a clone run; prepare a new authorized run configuration instead. Large file SHA256 receipts and the original local paths are recorded in the experiment snapshots.

`requirement.txt` and `requirements_win_no_box2d.txt` preserve the existing dependency lists. The accepted O1/V11 environment uses Python 3.8, PyTorch 1.11 and the previously validated MuJoCo/Gym setup. This upload installs no packages and starts no training.

## Git workflow

The source files are preserved byte-for-byte (`.gitattributes: * -text`) to retain scientific hashes. Before each update, inspect `git status` and `git diff`, commit only the intended code/configuration, then push to `origin/main`. Commit new work on a branch when appropriate. Keep local datasets, credentials, model weights and environments outside Git.

The repository's initial publication inventory and SHA256 digests are under `experiments/UPLOAD_MANIFEST_20261102.json`. Full raw logs and model artifacts remain local.
