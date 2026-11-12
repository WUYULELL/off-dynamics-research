"""Read-only GPU admission and conservative classification of external interruptions."""
import subprocess

from common import compute_process_rows, gpu_info, nvidia_read, pmon_process_rows, utc


def snapshot(index):
    value = gpu_info(index)
    compute = nvidia_read(['--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'])
    pmon = nvidia_read(['pmon','-i',str(index),'-c','1'])
    selected = [row for row in compute_process_rows(compute['stdout']) if row['gpu_uuid'] == value['uuid']]
    visible = sorted({row['pid'] for row in selected+pmon_process_rows(pmon['stdout'],index)})
    return dict(time=utc(),gpu=value,visible_pids=visible,compute=selected,pmon=pmon['stdout'])


def reason(sample, expected, policy):
    value = sample['gpu']
    if (value['uuid'],value['driver']) != (expected['uuid'],expected['driver']):
        return 'physical device identity or driver changed'
    if sample['visible_pids']:
        return 'a process is visible on this GPU'
    if value['utilization_gpu_percent'] != policy['idle_utilization_percent']:
        return 'nonzero GPU utilization'
    if value['memory_used_MiB'] > policy['resident_memory_allowance_MiB']:
        return 'unexplained occupied GPU memory'
    return None


def dmesg_tail():
    try:
        p = subprocess.run(['dmesg','--level=err,warn'],text=True,capture_output=True,timeout=5)
        return dict(returncode=p.returncode,lines=(p.stdout or p.stderr).splitlines()[-20:])
    except (OSError,subprocess.TimeoutExpired) as error:
        return dict(error=str(error))


def external_reason(code, failure, device, external_pids, current):
    if failure is None and code in (-9,-15):
        return 'external SIGKILL' if code == -9 else 'external SIGTERM'
    if failure is None or device is None or failure['type'] != 'RuntimeError':
        return None
    msg = failure['message'].lower()
    if ('cuda out of memory' in msg or 'cuda error: out of memory' in msg) and external_pids:
        return 'CUDA out of memory with externally occupied device'
    unavailable = (current is not None and current.get('error'))
    cuda_lost = any(key in msg for key in ('cuda error','cuda driver','cuda-capable device','nvml'))
    if cuda_lost and (unavailable or 'fallen off the bus' in msg or 'gpu is lost' in msg):
        return 'CUDA device became unavailable'
    return None
