"""Explicit, offline-only export bridge for the locked OTDF Walker2d task.

No work happens on import. --verify-only uses only the Python standard library.
The actual export constructs Gym environments but never calls step/rollout/train.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import sys
from datetime import datetime, timezone

SETUP = Path(__file__).resolve().parent
REPO = SETUP.parent / "OTDF"
TASK = "walker2d-morph-srcdatatype-medium-tardatatype-medium"
DEFAULT_INPUTS = SETUP / "artifacts" / TASK
COMMIT = "41e2e626844bd6b65e2316a418ce4d6df4bbe775"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path):
    path = Path(path).resolve()
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}


def verify_record(record):
    path = Path(record["path"])
    if not path.is_file() or path.stat().st_size != record["bytes"]:
        raise RuntimeError("Missing or changed file: {}".format(path))
    if sha256(path) != record["sha256"]:
        raise RuntimeError("SHA256 mismatch: {}".format(path))


def verify_inputs(inputs, require_cost=False):
    manifest_path = Path(inputs) / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["task"] != TASK or manifest["commit"] != COMMIT:
        raise RuntimeError("Wrong task or source revision in manifest")
    for record in manifest["original_files"] + list(manifest["features"].values()):
        verify_record(record)
    if require_cost:
        cost_path = REPO / "costlogs" / (TASK + ".hdf5")
        metadata = json.loads(cost_path.with_suffix(".manifest.json").read_text(encoding="utf-8"))
        if metadata["inputs_manifest_sha256"] != sha256(manifest_path):
            raise RuntimeError("Cost was generated for a different input manifest")
        if metadata["source_rows"] != manifest["features"]["source"]["rows"]:
            raise RuntimeError("Source/cost row count mismatch")
        verify_record(metadata["cost_file"])
    return manifest


def configure_environment(dataset_dir, mujoco_dir):
    for name in list(os.environ):
        if name.upper().startswith("ODRL_"):
            del os.environ[name]
    os.environ["D4RL_DATASET_DIR"] = str(Path(dataset_dir).resolve())
    os.environ["D4RL_SUPPRESS_IMPORT_ERROR"] = "1"
    os.environ["MUJOCO_PY_MUJOCO_PATH"] = str(Path(mujoco_dir).resolve())
    # D4RL imports optional dm_control kitchen tasks. Disable their unused GL
    # renderer so GLFW desktop initialization cannot block headless training.
    os.environ["MUJOCO_GL"] = "disable"
    prefixes = [str(Path(mujoco_dir) / "bin"),
                "D:/ANACONDA/envs/offdynamics/Library/bin",
                "D:/ANACONDA/envs/offdynamics",
                "D:/ANACONDA/envs/offdynamics/Scripts"]
    os.environ["PATH"] = os.pathsep.join(prefixes + [os.environ.get("PATH", "")])


def prepare_official_import_path():
    # The parent ODRL project has a regular package named algo, while the official
    # OTDF algo directory is a namespace package. Exclude that parent root.
    project_root = REPO.parent.parent.resolve()
    os.chdir(str(REPO))
    sys.path[:] = [entry for entry in sys.path if Path(entry or os.getcwd()).resolve() != project_root]
    sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, default=DEFAULT_INPUTS)
    parser.add_argument("--dataset-dir", type=Path, default=Path("C:/Users/TXin/.d4rl/datasets"))
    parser.add_argument("--mujoco-dir", type=Path, default=Path("C:/Users/TXin/.mujoco210/mujoco210"))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--require-cost", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        manifest = verify_inputs(args.inputs, args.require_cost)
        if Path(manifest["original_files"][0]["path"]).resolve() != (args.dataset_dir / "walker2d_medium-v2.hdf5").resolve():
            raise RuntimeError("Selected D4RL cache differs from the exported source cache")
        print("Verified dataset/code/feature hashes; source rows={}, target rows={}".format(
            manifest["features"]["source"]["rows"], manifest["features"]["target"]["rows"]))
        return
    if args.require_cost:
        parser.error("--require-cost must be used with --verify-only")
    if args.inputs.exists():
        raise FileExistsError("Refusing to overwrite export directory: {}".format(args.inputs))
    source_file = args.dataset_dir / "walker2d_medium-v2.hdf5"
    target_file = REPO / "datasets" / "walker2d_morph_medium.hdf5"
    if not source_file.is_file() or not target_file.is_file():
        raise FileNotFoundError("The source cache and official target data must already exist; no download is performed")
    configure_environment(args.dataset_dir, args.mujoco_dir)
    prepare_official_import_path()
    import numpy as np
    import gym
    import d4rl
    import algo.utils as utils

    # Match the official loader and the unmodified ReplayBuffer.convert_D4RL.
    source_env = gym.make("walker2d-medium-v2")
    try:
        actual_source = Path(source_env.dataset_filepath).resolve()
        if actual_source != source_file.resolve():
            raise RuntimeError("D4RL resolved an unexpected source dataset: {}".format(actual_source))
        source = d4rl.qlearning_dataset(source_env)
        target = utils.call_tar_dataset("walker2d-morph", "medium")
    finally:
        source_env.close()
    source_features = np.hstack([source["observations"], source["actions"], source["next_observations"]])
    target_features = np.hstack([target["observations"], target["actions"], target["next_observations"]])
    if source_features.shape[1] != target_features.shape[1]:
        raise RuntimeError("Source/target feature dimensions differ")
    if min(len(source_features), len(target_features)) < 1:
        raise RuntimeError("Empty dataset")
    tracked = [source_file, target_file, REPO / "make_otdf_cost.py",
               REPO / "train_otdf.py", REPO / "algo" / "OTDF.py", REPO / "algo" / "utils.py",
               REPO / "config" / "otdf" / "walker2d_morph.yaml",
               REPO / "envs" / "common.py", REPO / "envs" / "assets" / "walker_morph.xml",
               Path(d4rl.__file__), Path(d4rl.__file__).parent / "offline_env.py"]
    manifest = {"schema": 1, "task": TASK, "commit": COMMIT,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "python": platform.python_version(), "numpy": np.__version__, "gym": gym.__version__,
                "feature_order": ["observations", "actions", "next_observations"],
                "loader_note": "Official qlearning_dataset and call_tar_dataset; no row sorting, normalization or boundary fixes",
                "original_files": [file_record(path) for path in tracked], "features": {}}
    args.inputs.mkdir(parents=True, exist_ok=False)
    for name, array in [("source", source_features), ("target", target_features)]:
        path = args.inputs / (name + ".npy")
        with path.open("xb") as handle:
            np.save(handle, array, allow_pickle=False)
        record = file_record(path)
        record.update(rows=int(array.shape[0]), columns=int(array.shape[1]), dtype=str(array.dtype))
        manifest["features"][name] = record
    with (args.inputs / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
    print("Export complete: {}".format(args.inputs))
    print("Rows: source={}, target={}; no OT or learning was run".format(len(source_features), len(target_features)))


if __name__ == "__main__":
    main()
