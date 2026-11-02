"""Process-local bootstrap for the unmodified official OTDF training entry point.

Importing this module only defines functions. main() runs training and must only
be called after the user explicitly authorizes Smoke or Train.
"""
import argparse
import os
from pathlib import Path
import runpy
import sys

from export_ot_inputs import REPO, configure_environment, prepare_official_import_path


def configure_runtime(dataset_dir, mujoco_dir):
    """Set paths inside Python, without importing any experiment libraries."""
    for name in ("PYTHONPATH", "PYTHONHOME"):
        os.environ.pop(name, None)
    # pathlib uses Windows-native backslashes for mujoco_py's exact PATH check.
    configure_environment(Path(dataset_dir).resolve(), Path(mujoco_dir).resolve())
    prepare_official_import_path()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--mujoco-dir", type=Path, required=True)
    parser.add_argument("official_args", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    official_args = args.official_args
    if official_args[:1] == ["--"]:
        official_args = official_args[1:]
    if not official_args:
        parser.error("Pass explicit official training arguments after --")
    configure_runtime(args.dataset_dir, args.mujoco_dir)
    entrypoint = REPO / "train_otdf.py"
    sys.argv = [str(entrypoint)] + official_args
    runpy.run_path(str(entrypoint), run_name="__main__")


if __name__ == "__main__":
    main()
