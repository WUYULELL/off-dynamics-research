"""Create explicit jobs; no training or GPU access."""
from common_v5 import *

def main():
    jobs=[]
    for stack,methods,seeds,steps,out,python,entry in [
      ('A',A_METHODS,range(5),500000,OUT,PY_A,HERE/'train_entry_v5.py'),
      ('B',B_METHODS,range(1,6),1000000,B_OUT,PY_B,B_CODE/'train_entry_v5.py')]:
        for kind in ('smoke','formal'):
            arms=(('ORACLE-W1',)+methods if stack=='A' else ('OTDF',)+methods) if kind=='smoke' else methods
            for i,method in enumerate(arms):
                for seed in ([min(seeds)] if kind=='smoke' else seeds):
                    folder=out/('s1' if kind=='smoke' else 'f1')/(str(i)+'_'+str(seed))
                    jobs.append(dict(id=stack+'_'+kind+'_'+str(i)+'_'+str(seed),stack=stack,kind=kind,phase='standalone' if kind=='smoke' else 'formal',
                      method=method,seed=seed,steps=5000 if kind=='smoke' else steps,out=str(folder),python=str(python),entry=str(entry)))
        standalone=next(j for j in jobs if j['stack']==stack and j['kind']=='smoke' and j['method']==methods[0])
        paired=dict(standalone,id=stack+'_concurrent',phase='concurrent',out=str(out/'s_parallel1/0'))
        jobs.append(paired)
    assert sum(j['kind']=='formal' for j in jobs)==30
    write(OUT/'training_plan.json',dict(protocol_sha256=sha(OUT/'protocol.json'),jobs=jobs))
    print('Plan ready: 30 formal, 8 standalone smoke, 2 concurrent smoke; awaiting GPU confirmation')

if __name__=='__main__':main()
