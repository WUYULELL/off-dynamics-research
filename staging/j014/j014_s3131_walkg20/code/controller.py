"""Controller entry point of every adaptive-source job (j013-j016): reads config.json and runs the controller of its
kind (preflight / train / dyn / select / simcheck). Writes only under this job's code/ folder; outputs/ holds the
retrieved results (no absolute paths, no host or user names). Exit code 0 only when the job's checks pass."""
import importlib
from pathlib import Path
import sys
import traceback

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def main():
    from asrc_common import read_json, scrub, utc, write_json
    cfg = read_json(HERE/'config.json')
    (HERE/'outputs').mkdir(exist_ok=True)
    try:
        module = importlib.import_module('ctl_'+cfg['kind'])
        code = module.main(cfg)
    except BaseException:
        text = scrub(traceback.format_exc())
        print(text, flush=True)
        try:
            write_json(HERE/'outputs'/'STATUS.json', dict(time=utc(), job=cfg.get('job'), phase='controller_error', traceback=text[-4000:]))
        except Exception:
            pass
        code = 1
    return code


if __name__ == '__main__':
    sys.exit(main())
