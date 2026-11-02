import os
import sys

if sys.version_info[:2] != (3, 8):
    print(
        "This installer must run under Python 3.8, "
        f"but got {sys.version.split()[0]}."
    )
    raise SystemExit(1)

if os.environ.get("CONDA_DEFAULT_ENV") != "offdynamics":
    print(
        "Run `conda activate offdynamics` first. "
        f"Current CONDA_DEFAULT_ENV={os.environ.get('CONDA_DEFAULT_ENV')!r}."
    )
    raise SystemExit(1)

print("Environment check: OK")
