"""Standard-library-only package check; no scientific imports, no execution."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
freeze = json.loads((root/'FROZEN.json').read_text(encoding='utf-8'))
for name, expected in freeze['files'].items():
    p = (root/name).resolve()
    assert root in p.parents and p.is_file(), name
    assert hashlib.sha256(p.read_bytes()).hexdigest() == expected, name
print(json.dumps(dict(passed=True, files=len(freeze['files']), training_updates=0, environment_steps=0, GPU_jobs=0), indent=2))
