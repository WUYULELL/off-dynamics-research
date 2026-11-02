"""Read-only milestones and a request to defer B at a safe job boundary."""
from common_v5 import *

def milestones():
    jobs=read(OUT/'training_plan.json')['jobs'];formal=[j for j in jobs if j['kind']=='formal']
    first=[j for j in formal if j['method'] in (A_METHODS[0],B_METHODS[0])]
    for label,required in [('interim',first),('final',formal)]:
        path=OUT/(label+'_ready.json')
        if not path.exists() and all((Path(j['out'])/'audit.json').exists() for j in required):
            # Different queues may reach the milestone together; replace only identical readiness state.
            write(path,dict(ready=True,job_ids=[j['id'] for j in required],delivery='pending'),replace=True)
    baseline=OUT/'smoke_audited.json'
    if baseline.exists():
        rates=read(baseline)['standalone_rates'];current=[]
        for stack,folder in [('A',OUT),('B',B_OUT)]:
            state=folder/'queue_status.json'
            if not state.exists():continue
            s=read(state)
            if s.get('status')!='running' or s.get('kind')!='formal':continue
            job=next(j for j in jobs if j['id']==s['current']);progress=Path(job['out'])/'progress.json'
            if progress.exists():current.append((job,read(progress)))
        if len(current)==2 and any(p['step']>=10000 and p['steps_per_second']<.5*rates[j['method']] for j,p in current):
            gate=OUT/'defer_B_until_A10.json'
            if not gate.exists():write(gate,dict(reason='Observed training throughput below half of corresponding standalone smoke',evidence=current,action='Defer next B job; preserve active job, prioritize A first ten'))
    return dict(completed=sum((Path(j['out'])/'audit.json').exists() for j in formal),total=30)

if __name__=='__main__':print(milestones())
