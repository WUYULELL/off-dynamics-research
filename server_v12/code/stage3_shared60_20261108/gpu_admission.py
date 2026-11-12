"""Conservative admission for the registered cards on this host."""
import json
import subprocess
import time

from stage3_setup import CAP, GPU_UUIDS


POLL_SECONDS = 60
SAMPLES = 3
EMPTY_MEMORY_MIB = 3
SHARED_UTIL_LIMIT = 60
WORKER_MEMORY_MIB = 4096
RESERVE_MEMORY_MIB = 8192
MAX_UNACCOUNTED_MIB = 256


def capacity(gpu, compute, pmon):
    pids = sorted({p['pid'] for p in compute} | {p['pid'] for p in pmon})
    used, total = gpu['memory_used_MiB'], gpu['memory_total_MiB']
    util = gpu['utilization_gpu_percent']
    assert 0 <= used <= total and 0 <= util <= 100
    if not pids:
        if used > EMPTY_MEMORY_MIB or util != 0:
            return dict(slots=0, reason='GPU activity without an identifiable process', pids=pids)
    else:
        if len(pids) >= CAP:
            return dict(slots=0, reason='all three process slots are occupied', pids=pids)
        memory_by_pid = {p['pid']: int(p['memory_raw']) for p in compute if p['memory_raw'].isdigit()}
        if set(memory_by_pid) != set(pids) or abs(used - sum(memory_by_pid.values())) > MAX_UNACCOUNTED_MIB:
            return dict(slots=0, reason='incomplete per-process memory accounting', pids=pids)
        if util > SHARED_UTIL_LIMIT:
            return dict(slots=0, reason='shared card utilization exceeds limit', pids=pids)
    headroom = max(0, (total - used - RESERVE_MEMORY_MIB) // WORKER_MEMORY_MIB)
    slots = min(1 if pids else CAP, CAP - len(pids), headroom)
    return dict(slots=slots, reason='shared' if slots and pids else 'idle' if slots else 'insufficient memory headroom', pids=pids)


def observe(index, common, driver):
    try:
        gpu = common.gpu_info(GPU_UUIDS[index])
        assert gpu['index'] == index and gpu['uuid'] == GPU_UUIDS[index] and gpu['driver'] == driver
        pmon = common.nvidia_read(['pmon', '-i', str(index), '-c', '1'])
        process_query = common.nvidia_read(['--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory',
                                            '--format=csv,noheader,nounits'])
        compute = [p for p in common.compute_process_rows(process_query['stdout']) if p['gpu_uuid'] == gpu['uuid']]
        monitored = common.pmon_process_rows(pmon['stdout'], index)
        return dict(index=index, gpu=gpu, compute=compute, pmon=monitored, **capacity(gpu, compute, monitored))
    except (AssertionError, OSError, subprocess.SubprocessError, ValueError) as error:
        return dict(index=index, slots=0, reason=f'GPU identity or visibility could not be verified: {error}', pids=[])


def sample(index, common, driver):
    rows = []
    for _ in range(SAMPLES):
        rows.append(observe(index, common, driver))
        if rows[-1]['slots'] == 0:
            break
        if len(rows) < SAMPLES:
            time.sleep(1)
    slots = min(row['slots'] for row in rows)
    consistent_pids = all(row['pids'] == rows[0]['pids'] for row in rows)
    result = dict(rows[-1], slots=slots if consistent_pids else 0, samples=rows)
    if not consistent_pids:
        result['reason'] = 'process roster changed during admission'
    return result


def rank(row):
    gpu = row['gpu']
    return bool(row['pids']), len(row['pids']), gpu['utilization_gpu_percent'], gpu['memory_used_MiB'], row['index']


def candidates(indices, common, driver):
    rows = [observe(index, common, driver) for index in indices]
    return sorted((row for row in rows if row['slots']), key=rank), rows


def wait(indices, label, common, driver, out=None, requested=1, processes=(), before_poll=None):
    started = time.monotonic()
    polls = 0
    while True:
        if before_poll is not None:
            before_poll()
        assert all(p.poll() is None for p in processes), 'A staged child exited before GPU release'
        ordered, rows = candidates(indices, common, driver)
        for candidate in ordered:
            stable = sample(candidate['index'], common, driver)
            if stable['slots'] >= requested:
                elapsed = time.monotonic() - started
                receipt = dict(label=label, blocked_polls=polls, elapsed_seconds=elapsed,
                               blocked_wait_seconds=elapsed if polls else 0., time=common.utc(), **stable)
                if out is not None:
                    common.write(out / 'admissions' / (label + '.json'), receipt)
                return receipt
            rows.append(stable)
        polls += 1
        info = dict(label=label, time=common.utc(), wait_seconds=time.monotonic() - started, candidates=rows)
        if out is not None:
            with (out / 'GPU_WAIT.jsonl').open('a', encoding='utf-8') as log:
                log.write(json.dumps(info, allow_nan=False) + '\n')
        print(f'{label}: waiting {POLL_SECONDS} seconds for {requested} slot(s)',
              [(row['index'], row['slots'], row['reason']) for row in rows], flush=True)
        time.sleep(POLL_SECONDS)
