"""Static check of the generated adaptive-source job directories (run from the repository root).

* every file listed in FROZEN.json has its SHA-256 and every file under code/ is listed;
* shared files are byte-identical to methods/adaptive_source/src and the shared digest is the same in all jobs;
* official OTDF / d4rl files have the expected official bytes;
* no CRLF in text files (the relay converts CRLF to LF when packing, which would break the hashes);
* no user / host names, IP addresses or e-mail addresses in any text file.
usage: python methods/adaptive_source/tests/verify_jobs.py
"""
import hashlib
import json
from pathlib import Path
import re
import sys

REPO = Path(__file__).resolve().parents[3]
SRC = REPO/'methods'/'adaptive_source'/'src'
sys.path.insert(0, str(SRC))
from asrc_common import D4RL_SHA256, OTDF_OFFICIAL_SHA256  # noqa: E402

TEXT = {'.py', '.json', '.md', '.sh', '.yaml', '.yml', '.txt', '.csv'}
BAD = re.compile(r'(C:\\\\Users|C:/Users|/home/[A-Za-z]|\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b|[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.(com|cn|edu|org|net)\b)')


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    dirs = sorted([d for d in (REPO/'jobs').glob('j01[3-6]*') if (d/'code'/'FROZEN.json').is_file()] +
                  [d for d in (REPO/'staging').rglob('j01*') if (d/'code'/'FROZEN.json').is_file()])
    shared = {str(p.relative_to(SRC)): sha(p) for p in SRC.rglob('*') if p.is_file() and '__pycache__' not in p.parts}
    digests, problems = set(), []
    for d in dirs:
        code = d/'code'
        fz = json.loads((code/'FROZEN.json').read_text(encoding='utf-8'))
        listed = fz['files']
        present = {str(p.relative_to(code)) for p in code.rglob('*') if p.is_file() and p.name != 'FROZEN.json'}
        if set(listed) != present:
            problems.append('%s: FROZEN list != files (%s)' % (d.name, sorted(set(listed) ^ present)[:5]))
        for n, h in listed.items():
            if sha(code/n) != h:
                problems.append('%s: hash mismatch %s' % (d.name, n))
        for n, h in shared.items():
            if listed.get(n) != h:
                problems.append('%s: shared file differs %s' % (d.name, n))
        for n, h in list(OTDF_OFFICIAL_SHA256.items())+list(D4RL_SHA256.items()):
            if listed.get(n) != h:
                problems.append('%s: official file %s' % (d.name, n))
        digests.add(fz['shared_files_digest'])
        for n in listed:
            p = code/n
            if p.suffix in TEXT:
                data = p.read_bytes()
                if b'\r\n' in data:
                    problems.append('%s: CRLF in %s' % (d.name, n))
                m = BAD.search(data.decode('utf-8', 'replace'))
                if m and not n.startswith(('d4rl/', 'otdf_official/')):
                    problems.append('%s: suspicious string %r in %s' % (d.name, m.group(0), n))
        job = json.loads((d/'job.json').read_text(encoding='utf-8'))
        if job['job_id'] != d.name or job.get('approved_by_user') is not False or float(job['max_hours']) > 24:
            problems.append('%s: job.json fields' % d.name)
    if len(digests) != 1:
        problems.append('shared digests differ: %s' % digests)
    print(json.dumps(dict(jobs=len(dirs), shared_files_digest=sorted(digests), problems=problems), indent=2))
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
