from common_v5 import *

def main():
    fresh(OUT);fresh(B_OUT)
    request=Path('D:/claude 离线/给Code的指令_v5_路径归因与OTDF对照.md')
    (OUT/'USER_SPEC_v5.md').write_bytes(request.read_bytes())
    protocol=dict(created=time.time(),spec_sha256=sha(request),task='walker2d-morph',
      A=dict(methods=A_METHODS,seeds=list(range(5)),steps=500000,smoke_steps=5000,
        draws=[64,64],source_weight=1.,loader='explicit_next_v3',target_n=5000,
        last5_steps=list(range(460000,500001,10000)),episodes=10,
        comparisons={'QA1':['CRITIC-ORACLE','TARGET-64'],'QA2':['CRITIC-ORACLE','ORACLE-W1'],
          'QA3':['CRITIC-ORACLE','CRITIC-SOURCE'],'QA4':['Q-ORACLE','CRITIC-ORACLE']},
        descriptive=[['CRITIC-ORACLE','TARGET'],['CRITIC-SOURCE','TARGET-64']]),
      B=dict(methods=B_METHODS,seeds=list(range(1,6)),steps=1000000,smoke_steps=5000,
        vae_steps_if_reg=10000,loader='official_OTDF_unchanged',target_n=4999,
        target_only_draws=[0,128],otdf_draws=[128,128],otdf_source_retained=102,
        proportion=.8,reg_weight=.5,weight=True,last5_steps=list(range(960000,1000001,10000)),episodes=10,
        comparisons={'QB1':['OTDF','IQL-TARGET'],'QB2':['OTDF','IQL-TARGET+REG'],
          'QB3':['IQL-TARGET+REG','IQL-TARGET'],'QB4':['OTDF','OTDF-NOREG']}),
      gate='paired mean > 2 * sample_SD / sqrt(5) and >=4 positive seeds; engineering criterion, not 95% CI',
      normalization={'min':10.079455055289959,'max':4398.43},
      reuse_A=str(V4_OUT/'s1/w0'),reuse_B=str(ROOT/'reproductions/otdf_setup/study_five_seeds.json'),
      tolerances=dict(A_parameters={'rtol':1e-5,'atol':1e-6},A_early_losses={'rtol':1e-4,'atol':1e-5},
        A_normalized_eval_abs=.001,B_initial_return_abs=.001,B_first_vae={'rtol':1e-4,'atol':1e-5}),
      scientific_review=[
        'QA2 identifies the intervention removing direct source actor loss, not proof it is the unique or main harmful mechanism.',
        'Source still changes actor indirectly through learned Q and V across updates; one-step invariance only tests the direct path.',
        'QA3 is the measured effect of this privileged label intervention, never a theoretical ceiling.',
        'Failure to pass a positive gate is not equivalence, proof of no value, or necessity of a component.',
        'QB2 adds filtered weighted source and changes total batch to 230; it identifies the full OTDF source-use package.',
        'B preserves official actor broadcasting behavior and the 4999-row next-observation loader; A and B are separate protocols.',
        'B pretraining/no-pretraining consumes different RNG streams; same seed does not imply equal subsequent sample sequences.',
        'No outcomes or signs are assumed from the specification; all planned comparisons will be reported.'],
      approval='CPU preparation authorized. GPU smoke needs confirmation; formal queues need the post-smoke confirmation specified by user document.',
      execution=dict(cpu_threads=2,queue_order='group then seed; A and B parallel only after throughput test',
        intermediate='deliver QA1 QA2 QB1 after the respective first groups complete; continue authorized remaining groups',
        slowdown='If concurrent throughput <50% standalone, do not kill active training: defer next B job at safe job boundary; retain completed and running evidence',
        retry='new attempt from zero, same seed, failed evidence retained, no return-based selection'))
    write(OUT/'protocol.json',protocol);write(B_OUT/'protocol_reference.json',dict(path=str(OUT/'protocol.json'),sha256=sha(OUT/'protocol.json')))
    protected=set(Path(p) for p in read(V4_OUT/'protected_manifest.json')['sha256'])
    for base in ('oracle_diag','reproductions/OTDF','reproductions/otdf_setup'):
        for p in (ROOT/base).rglob('*'):
            if HERE in p.parents or OUT in p.parents:continue
            if os.path.isfile(winpath(p)):protected.add(p)
    hashes={}
    for i,p in enumerate(sorted(protected)):
        hashes[str(p)]=sha(p)
        if i%500==0:print('Hash inventory:',i,flush=True)
    write(OUT/'protected_manifest.json',dict(count=len(hashes),sha256=hashes))
    write(OUT/'active_execution.json',dict(state='cpu_preparation',gpu_authorized=False,queue_running=False))
    print('PROTOCOL FROZEN',len(hashes),flush=True)

if __name__=='__main__':main()
