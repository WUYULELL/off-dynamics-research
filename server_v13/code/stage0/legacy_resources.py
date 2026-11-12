"""Read-only Linux resource measurements and precommitted engineering limits."""
import math
import os
from pathlib import Path
import subprocess
import time

GIB=1024**3


def key_values(path):
    result={}
    try:
        for line in Path(path).read_text().splitlines():
            p=line.split()
            if len(p)>=2 and p[1].isdigit():result[p[0].rstrip(':')]=int(p[1])
    except OSError:pass
    return result


def cgroup_roots():
    records=[]
    membership=Path('/proc/self/cgroup').read_text()
    mounts=Path('/proc/self/mountinfo').read_text()
    group=next((s.split(':',2)[2] for s in membership.splitlines() if s.startswith('0::')),None)
    for line in mounts.splitlines():
        front,sep,back=line.partition(' - ')
        if not sep or back.split()[0]!='cgroup2':continue
        fields=front.split();mount=Path(fields[4]);mount_root=fields[3]
        if group is None:continue
        if group=='/':target=mount
        elif mount_root=='/':target=mount/group.lstrip('/')
        elif group==mount_root or group.startswith(mount_root.rstrip('/')+'/'):
            target=mount/group[len(mount_root):].lstrip('/')
        else:target=mount
        target=target.resolve();mount=mount.resolve()
        if target!=mount and mount not in target.parents:target=mount
        if not target.is_dir():target=mount
        while True:
            records.append(target)
            if target==mount:break
            target=target.parent
    if not records and Path('/sys/fs/cgroup/cpu.max').is_file():records=[Path('/sys/fs/cgroup')]
    return list(dict.fromkeys(records)),membership,mounts


def limits_snapshot():
    groups,membership,mounts=cgroup_roots()
    records=[];quotas=[];heads=[];unresolved=[]
    for path in groups:
        r={'path':str(path)}
        for key in ('cpu.max','cpu.stat','memory.max','memory.current','cpuset.cpus.effective'):
            p=path/key
            if p.is_file():
                try:r[key]=p.read_text().strip()
                except OSError:unresolved.append(str(p))
        if 'cpu.max' in r:
            q,t=r['cpu.max'].split()
            if q!='max':quotas.append(float(q)/float(t))
        if r.get('memory.max','max')!='max' and 'memory.current' in r:
            heads.append(max(0,int(r['memory.max'])-int(r['memory.current'])))
        records.append(r)
    # v1 fallback, recorded explicitly rather than assumed unlimited.
    if not records:
        q=Path('/sys/fs/cgroup/cpu/cpu.cfs_quota_us');period=q.with_name('cpu.cfs_period_us')
        if q.is_file() and period.is_file():
            a,b=int(q.read_text()),int(period.read_text())
            if a>0:quotas.append(a/b)
            records.append(dict(path=str(q.parent),cpu_cfs_quota_us=a,cpu_cfs_period_us=b))
        ml=Path('/sys/fs/cgroup/memory/memory.limit_in_bytes');mc=ml.with_name('memory.usage_in_bytes')
        if ml.is_file() and mc.is_file():heads.append(max(0,int(ml.read_text())-int(mc.read_text())))
    affinity=sorted(os.sched_getaffinity(0))
    nproc_env=os.environ.copy()
    for key in ('OMP_NUM_THREADS','OMP_THREAD_LIMIT'):nproc_env.pop(key,None)
    nproc=int(subprocess.check_output(['nproc'],env=nproc_env,text=True).strip())
    effective=min([len(affinity),nproc]+quotas)
    mem=key_values('/proc/meminfo');available=mem['MemAvailable']*1024
    if heads:available=min([available]+heads)
    return dict(nproc=nproc,nproc_queried_without_OMP_override=True,logical_cpu_count=os.cpu_count(),affinity=affinity,
        cpu_quota_cores=None if not quotas else min(quotas),effective_visible_cpu_cores=effective,
        cgroup_mapping_found=bool(records),cgroup_records=records,cgroup_membership=membership,
        cgroup_mountinfo=mounts,unreadable_fields=unresolved,
        host_memory=mem,effective_available_memory_bytes=available,
        cgroup_memory_headroom_bytes=None if not heads else min(heads))


def cpu_ticks(affinity=None):
    selected={int(x) for x in affinity} if affinity is not None else None
    total=idle=0
    for line in Path('/proc/stat').read_text().splitlines():
        parts=line.split()
        if not parts or not parts[0].startswith('cpu'):continue
        if selected is None and parts[0]!='cpu':continue
        if selected is not None and (parts[0]=='cpu' or int(parts[0][3:]) not in selected):continue
        values=[int(x) for x in parts[1:9]] # guest times are already included; do not double-count.
        total+=sum(values);idle+=values[3]+values[4]
    return dict(total=total,idle=idle)


def busy_percent(previous,current):
    total=current['total']-previous['total'];idle=current['idle']-previous['idle']
    return None if total<=0 else 100*max(0,total-idle)/total


def owned_process(pid):
    try:
        text=Path('/proc/%d/stat'%pid).read_text();tail=text[text.rfind(')')+2:].split()
        status=key_values('/proc/%d/status'%pid)
        return dict(pid=pid,start_ticks=int(tail[19]),cpu_ticks=int(tail[11])+int(tail[12]),
            RSS_bytes=status.get('VmRSS',0)*1024,peak_RSS_bytes=status.get('VmHWM',0)*1024)
    except (OSError,ValueError,IndexError):return None


def tier_allowed(n,limits,policy,per_worker_RSS=None,per_worker_gpu_MiB=None,gpu_total_MiB=None):
    reasons=[]
    effective=limits['effective_visible_cpu_cores']
    cap=max(1,int(math.floor(effective))-policy['reserve_cpu_cores'])
    if not limits['cgroup_mapping_found'] and n>1:reasons.append('CPU cgroup limits are unresolved; only single-process measurement is permitted')
    if n>min(6,cap):reasons.append('Concurrency exceeds visible CPU quota/affinity with reserved CPU headroom')
    rss=policy['initial_worker_memory_estimate_bytes'] if per_worker_RSS is None else per_worker_RSS
    required=policy['reserve_memory_bytes']+policy['memory_prediction_multiplier']*n*rss
    if limits['effective_available_memory_bytes']<required:reasons.append('Insufficient observed memory headroom for the next tier')
    if per_worker_gpu_MiB is not None and gpu_total_MiB:
        if n*per_worker_gpu_MiB>policy['max_projected_GPU_memory_fraction']*gpu_total_MiB:
            reasons.append('Projected GPU memory footprint exceeds the precommitted fraction')
    return reasons


def group_rates(completions,release_monotonic):
    total=sum(c['training_updates'] for c in completions)
    wall=max(c['train_end_monotonic'] for c in completions)-release_monotonic
    assert wall>0
    return dict(total_training_updates=total,group_training_wall_seconds=wall,
        aggregate_group_updates_per_second=total/wall,
        sum_individual_updates_per_second=sum(c['training_updates']/c['training_seconds'] for c in completions),
        per_process_updates_per_second=[c['training_updates']/c['training_seconds'] for c in completions])


def dynamic_snapshot(limits):
    mem=key_values('/proc/meminfo');available=mem['MemAvailable']*1024;stats={}
    for entry in limits['cgroup_records']:
        p=Path(entry['path'])
        if (p/'memory.max').is_file() and (p/'memory.current').is_file():
            maximum=(p/'memory.max').read_text().strip()
            if maximum!='max':available=min(available,max(0,int(maximum)-int((p/'memory.current').read_text())))
        if (p/'cpu.stat').is_file():stats[str(p)]=key_values(p/'cpu.stat')
    return dict(monotonic=time.monotonic(),host_cpu=cpu_ticks(),affinity_cpu=cpu_ticks(limits['affinity']),
                effective_available_memory_bytes=available,cgroup_cpu_stats=stats)


class Monitor:
    def __init__(self,limits,policy,gpu_reader):
        self.limits,self.policy,self.gpu_reader=limits,policy,gpu_reader
        self.previous=dynamic_snapshot(limits);self.process_previous={};self.pressure_since={};self.warnings=[]
        self.observed=[];self.max_worker_RSS=0

    def sample(self,processes,phase):
        x=dynamic_snapshot(self.limits);dt=x['monotonic']-self.previous['monotonic']
        gpu=self.gpu_reader();owned=[];cpu_percent=0.
        hz=os.sysconf('SC_CLK_TCK')
        for p in processes:
            if p.poll() is not None:continue
            rec=owned_process(p.pid)
            if rec is None:continue
            old=self.process_previous.get(p.pid)
            rec['cpu_percent_one_core_100']=None
            if old and old['start_ticks']==rec['start_ticks'] and dt>0:
                rec['cpu_percent_one_core_100']=max(0,rec['cpu_ticks']-old['cpu_ticks'])/hz/dt*100
                cpu_percent+=rec['cpu_percent_one_core_100']
            self.process_previous[p.pid]=rec
            self.max_worker_RSS=max(self.max_worker_RSS,rec['peak_RSS_bytes'])
            owned.append(rec)
        host=busy_percent(self.previous['host_cpu'],x['host_cpu'])
        affinity=busy_percent(self.previous['affinity_cpu'],x['affinity_cpu'])
        throttled=[]
        for path,stats in x['cgroup_cpu_stats'].items():
            old=self.previous['cgroup_cpu_stats'].get(path,{})
            periods=stats.get('nr_periods',0)-old.get('nr_periods',0)
            count=stats.get('nr_throttled',0)-old.get('nr_throttled',0)
            if periods>0:throttled.append(max(0,count)/periods)
        rate=max(throttled) if throttled else None
        flags=dict(high_CPU=(host is not None and host>self.policy['CPU_busy_percent']) or
                   (affinity is not None and affinity>self.policy['CPU_busy_percent']),
                   throttled=rate is not None and rate>self.policy['max_throttled_period_fraction'],
                   low_memory=x['effective_available_memory_bytes']<self.policy['reserve_memory_bytes'])
        for name,active in flags.items():
            if not active:self.pressure_since.pop(name,None)
            else:
                self.pressure_since.setdefault(name,x['monotonic'])
                if x['monotonic']-self.pressure_since[name]>=self.policy['pressure_duration_seconds'] and name not in self.warnings:
                    self.warnings.append(name)
        assert x['effective_available_memory_bytes']>=self.policy['emergency_memory_floor_bytes'],'Emergency memory floor reached; stop owned workers'
        record=dict(monotonic=x['monotonic'],phase=phase,gpu=gpu,owned_processes=owned,
            owned_CPU_percent_one_core_100=cpu_percent,host_CPU_busy_percent=host,affinity_CPU_busy_percent=affinity,
            cgroup_throttled_period_fraction=rate,available_memory_bytes=x['effective_available_memory_bytes'],
            pressure_flags=flags,persistent_pressure_warnings=list(self.warnings))
        self.previous=x;self.observed.append(record)
        return record
