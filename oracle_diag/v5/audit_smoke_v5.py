"""CPU smoke/reuse/throughput gate. Never authorizes formal training itself."""
from audit_v5 import *

def main():
    jobs=[j for j in read(OUT/'training_plan.json')['jobs'] if j['kind']=='smoke'];assert len(jobs)==10
    rows={}
    for j in jobs:
        row=audit_job(j)
        if j['method']=='ORACLE-W1':row['reuse']=reuse_a(j['out'])
        if j['method']=='OTDF':row['reuse']=reuse_b(j['out'])
        rows[j['id']]=row
    standalone={j['method']:read(Path(j['out'])/'progress.json')['steps_per_second'] for j in jobs if j['phase']=='standalone'}
    ratios={j['stack']:read(Path(j['out'])/'progress.json')['steps_per_second']/standalone[j['method']] for j in jobs if j['phase']=='concurrent'}
    # Conservative projection uses measured end-to-end smoke wall time, not only GPU kernel time.
    wall_rates={};vae_seconds={}
    for j in jobs:
        if j['phase']!='standalone':continue
        folder=Path(j['out']);pretrain=read(folder/'vae_completed.json')['elapsed'] if (folder/'vae_completed.json').exists() else 0.
        wall_rates[j['method']]=j['steps']/(read(folder/'progress.json')['elapsed']-pretrain)
        vae_seconds[j['method']]=pretrain
    eta={stack:sum(5*(read(OUT/'protocol.json')[stack]['steps']/wall_rates[m]+vae_seconds[m])/min(1.,ratios[stack]) for m in methods)/3600
      for stack,methods in [('A',A_METHODS),('B',B_METHODS)]}
    count=protected_check()
    write(OUT/'smoke_audited.json',dict(passed=True,rows=rows,standalone_rates=standalone,concurrent_ratios=ratios,conservative_eta_hours=eta,
      B_over_72h=eta['B']>72,full_five_seed_plan_unchanged=True,formal_authorization='pending user post-smoke confirmation',protected_files=count))
    print('All GPU smoke evidence audited; formal authorization still pending')

if __name__=='__main__':main()
