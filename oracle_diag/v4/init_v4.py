"""Freeze scientific protocol and preserve hashes before v4 diagnostics."""
from support_v4 import *


def main():
    fresh(OUT)
    request = Path('D:/claude 离线/给Code的指令_v4_归因实验.md')
    (OUT / 'USER_SPEC_v4.md').write_bytes(request.read_bytes())
    protocol = dict(created=time.time(), request_sha256=sha(request),
        task='walker2d-morph', direction='normal D4RL medium -> shifted OTDF medium',
        loader='explicit_next_v3', methods=list(METHODS), seeds=list(range(5)),
        formal_steps=500000, smoke_steps=5000, evaluation_every=10000, episodes=10,
        last5_steps=[460000,470000,480000,490000,500000],
        source_n=999322, target_n=5000, gamma=.99, eta=1.,
        source_weights={'ORACLE-ONLY': 'unweighted batch mean', 'SOURCE-ONLY': 'unweighted batch mean',
                        'TARGET-64': 'no source', 'ORACLE-W1': 1.0},
        draw_counts=DRAW_COUNTS, oracle_receipt=str(oracle_receipt()), oracle_receipt_sha256=sha(oracle_receipt()),
        source_only='Frozen TargetIQL, routed source buffer; no shifted dataset or target empirical statistics',
        mixed='mean(target loss)+1.0*mean(source loss) for Q/V/actor; only source Q labels replaced',
        main_comparisons={'Q1':['ORACLE-ONLY','TARGET'], 'Q2':['ORACLE-ONLY','SOURCE-ONLY'],
                          'Q3':['TARGET','TARGET-64'], 'Q4':['ORACLE-W1','TARGET']},
        descriptive_comparisons=[['SOURCE-ONLY','TARGET'],['ORACLE-W1','MIX-ORACLE-BACKUP'],['TARGET-64','MIX-CONST']],
        gate='paired mean > 2*sample_SD/sqrt(5) and >=4/5 positive; engineering gate, not a 95% CI',
        target_reuse=dict(reference=str(V3_OUT/'s1/t0'), initialization='exact SHA256', first_indices='exact',
            config='exact effective learner config', model_at_5000=dict(rtol=1e-5,atol=1e-6),
            evaluation_absolute_score_tolerance=.001,
            early_losses='v3 did not save early losses; reconstruct 10 CPU updates with frozen v3 and v4, exact parity; compare new GPU trace to CPU with rtol=1e-4, atol=1e-5',
            failure='stop; do not reuse TARGET or automatically add 5 runs'),
        scientific_review=[
            'ORACLE-CONST near-zero mean is not equivalence; seed differences were large.',
            'Oracle cache is privileged simulator evidence, not independently known conditional truth.',
            'Q1 positivity supports this complete source-only intervention, not isolated state coverage.',
            'Q1/Q4 patterns do not uniquely attribute causality to weights or mixing; Q4 changes Q/V/actor weighting and total loss scale.',
            'Q1 failure is not proof of distribution mismatch or a performance upper bound.',
            'Q1/Q2 failure is not proof that all dynamics correction is useless.',
            'Q3 isolates target batch under target-only; it cannot decompose mixed-training loss causally without interaction assumptions.',
            'Old v3 files remain immutable; any interpretation supplement is written only in v4.'
        ],
        approval='CPU preparation authorized. Separate explicit confirmation before GPU smoke and before formal queue per sections A/B; no authorization fabricated.',
        queue_order='method first, seed 0..4; report after first 10 formal runs without pausing',
        restart='Preserve failed output; new attempt, same seed from step 0, no score-based retries',
        normalization=dict(min=10.079455055289959,max=4398.43),
        runtime_prior_hours=51.0, runtime_prior_note='20 * v3 mean 2.55h; source-only may be faster; revise after smoke')
    write(OUT/'protocol.json', protocol)
    protected = set()
    for base in ('dataset','envs','config','algo','oracle_diag'):
        for path in (ROOT/base).rglob('*'):
            if HERE in path.parents or OUT in path.parents:
                continue
            if os.path.isfile(winpath(path)):
                protected.add(path)
    # Existing root run scripts and the external data/cache assets are also protected.
    for path in ROOT.iterdir():
        if path.is_file() and path.suffix.lower() in ('.py','.ps1','.bat','.sh','.yaml','.json'):
            protected.add(path)
    for path in read(V3_OUT/'protected_manifest.json')['sha256']:
        protected.add(Path(path))
    hashes = {}
    for index, path in enumerate(sorted(protected)):
        hashes[str(path)] = sha(path)
        if index and index % 1000 == 0:
            print('Protected files frozen:', index, flush=True)
    write(OUT/'protected_manifest.json',dict(count=len(hashes),sha256=hashes,created=time.time()))
    write(OUT/'status.json',dict(state='cpu_preparation',gpu_authorized=False))
    print('V4 protocol frozen; protected files:',len(hashes),flush=True)


if __name__ == '__main__':
    main()
