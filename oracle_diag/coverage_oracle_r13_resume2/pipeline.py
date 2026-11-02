# -*- coding: utf-8 -*-
"""Only G205 remainder and five sealed-mask R runs; aggregate original20 and stop."""
from runtime import *
from decision import decide
import subprocess
import traceback

class Queue:
    def __init__(self):
        check_frozen()
        a = read(OUT / 'cpu_acceptance.json')
        c = read(OUT / 'CPU_ACCEPTANCE_COMMITMENT.json')
        assert a['passed'] and a['CPU_only'] and a['CPU_split_resume_bitwise_identical']
        assert sha(OUT / 'cpu_acceptance.json') == c['cpu_acceptance_sha256']
        assert sha(OUT / 'OVERLAP_LOSSES.json') == c['overlap_sha256']
        assert sha(OUT / 'code_frozen.json') == c['code_frozen_sha256']
        gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=uuid', '--format=csv,noheader'], text=True)
        assert read(OUT / 'recovery.json')['expected_GPU_uuid'] in gpu.split()
        write(OUT / 'queue_started.json', dict(pid=os.getpid(), time=time.time(), command=sys.argv))
        self.jobs = []
        self.results = []
        for r in read(OUT / 'reuse_receipts.json')['runs']:
            assert sha(r['audit']) == r['audit_sha256']
            a = read(r['audit']); assert a['passed']
            self.results.append(a)
        assert len(self.results) == 14
        self.status('STARTING')

    def status(self, stage, **kw):
        write(OUT / 'queue_status.json', dict(stage=stage, queue_pid=os.getpid(), updated=time.time(),
            dispatched_new_jobs=len(self.jobs), audited_jobs=len(self.results), reused_completed_jobs=14,
            total_jobs=20, total_new_GPU_jobs=6, C_grid_authorized=False, original_formal_authorized=False,
            task2_authorized=False, estimator_development_authorized=False, **kw), replace=True)

    def protect(self, label):
        entries = read(OUT / 'protected_manifest.json')['sha256']
        bad = [p for p, h in entries.items() if not os.path.exists(winpath(p)) or sha(p) != h]
        write(OUT / ('protected_' + label + '.json'), dict(passed=not bad, count=len(entries), changed=bad))
        assert not bad

    def child(self, script, args, name, cpu=False):
        command = [str(PYTHON), '-B', str(HERE / script)] + [str(a) for a in args]
        environ = os.environ.copy(); environ['CUDA_VISIBLE_DEVICES'] = '-1' if cpu else '0'
        environ['PYTHONDONTWRITEBYTECODE'] = '1'
        with open(OUT / 'logs' / (name + '.stdout.log'), 'xb') as stdout, open(OUT / 'logs' / (name + '.stderr.log'), 'xb') as stderr:
            p = subprocess.Popen(command, cwd=str(ROOT), env=environ, stdout=stdout, stderr=stderr,
                creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
            self.status(script, active_pid=p.pid, active_command=command, log_prefix=str(OUT / 'logs' / name))
            rc = p.wait()
        if rc:
            raise RuntimeError('%s exited%d; no automatic retraining' % (script, rc))

    def main(self):
        self.protect('before_gpu')
        for job in read(OUT / 'training_plan.json')['jobs']:
            check_job(job); self.jobs.append(job)
            write(OUT / 'dispatch_progress.json', dict(jobs=self.jobs), replace=True)
            script = 'resume_train.py' if job['arm'] == 'G' else 'train.py'
            self.child(script, ['--job', OUT / 'jobs' / (job['id'] + '.json')], job['id'])
            if job['arm'] == 'G':
                assert read(Path(job['out']) / 'OVERLAP_VERIFIED.json')['passed']
            self.child('audit.py', ['--out', job['out']], job['id'] + '_audit', cpu=True)
            a = read(Path(job['out']) / 'audit.json'); assert a['passed']
            t = next(r for r in self.results if r['job']['arm'] == 'T' and r['job']['seed'] == job['seed'])
            assert a['sampling']['target_stream_sha256'] == t['sampling']['target_stream_sha256']
            assert read(Path(job['out']) / 'initialization.json') == read(Path(t['job']['out']) / 'initialization.json')
            self.results.append(a); self.status('job_audited', last_audited=job['id'])
        assert len(self.results) == 20
        assert sorted((a['job']['arm'], a['job']['seed']) for a in self.results) == sorted((a, s) for a in ('T','U','G','R') for s in range(201,206))
        groups = {arm: [a for a in self.results if a['job']['arm'] == arm] for arm in ('T', 'U', 'G', 'R')}
        result = decide(groups)
        result['group_means'] = {arm: sum(a['Last5'] for a in values)/5 for arm, values in groups.items()}
        result['group_Final_means'] = {arm: sum(a['Final'] for a in values)/5 for arm, values in groups.items()}
        result['runs'] = groups; result['protocol_sha256'] = sha(OUT / 'protocol.json')
        result['recovery'] = dict(reused14=True, resumed_G205_from100k=True,
            original_results=str(PRIOR), original_scientific_protocol_unchanged=True)
        write(OUT / 'CONFIRMATION_RESULT.json', result)
        self.protect('stage_end')
        lines = ['# R13 独立种子确认报告（暂停后续跑）', '', '**' + result['conclusion'] + '。已停止。**', '',
            '原14项完整结果复用；G205从100k完整状态续跑、24项旧GPU损失严格相等，R五项从零训练且使用原封存掩码。',
            'R13是在看到R12验证后修订、确认结果前固定。小效应标签不是原≥5分开发成功；本轮无C，任何结果不触发估计器开发。', '',
            '|组别|Last-5五种子均值|Final均值|', '|---|---:|---:|']
        lines += ['|%s|%.5f|%.5f|' % (a, result['group_means'][a], result['group_Final_means'][a]) for a in ('T','U','G','R')]
        for metric in ('Last5','Final'):
            lines += ['', '## ' + metric, '', '|配对|均值|样本SD|SE|正号|工程判据|', '|---|---:|---:|---:|---:|---|']
            for name, s in result['paired_' + metric].items():
                lines += ['|%s|%.5f|%.5f|%.5f|%d/5|%s|' % (name, s['mean'], s['sample_SD'], s['SE'], s['positive'], s['passed'])]
        lines += ['', '仅Last-5进入三项预设判定，Final单独描述。三项全过才记小幅稳定挑选效应，任一未过结束方向二、转方向一，不追加训练。'] + result['notes']
        lines += ['', 'T温度为网格上界且三档单调上升；G−T在[3,7]时须列主要保留意见，本轮不因此追加T128。',
            '小样本工程门槛不等于严格95%统计结论；未过不证明无效或等效，不能推出所有source有害或收益上界。',
            '固定oracle标签和参考访问分布下的诊断结果，不直接与OTDF/TABB排名。未来重启方向二须新假设和新数据，不事后改门槛。',
            '交付前另在新目录CPU独立复算原始TB、逐episode回报、六项配对与结果树，并核对全部保护文件。']
        report = OUT / 'CONFIRMATION_REPORT_R13.md'
        with open(report, 'x', encoding='utf-8') as f:
            f.write('\n'.join(lines) + '\n')
        write(OUT / 'STAGE_COMPLETED.json', dict(status='AWAITING_USER_DECISION', completed_jobs=20,
            report=str(report), all_three_passed=result['all_three_passed'],
            estimator_development_triggered=False, no_additional_training=True, time=time.time()))
        self.status('AWAITING_USER_DECISION', report=str(report))

if __name__ == '__main__':
    q = None
    try:
        q = Queue(); q.main()
    except BaseException:
        path = OUT / ('queue_failure_%d.json' % int(time.time()))
        write(path, dict(pid=os.getpid(), time=time.time(), traceback=traceback.format_exc(),
            action='CPU diagnosis and report; no automatic retraining'))
        if q:
            q.status('FAILED', failure=str(path))
        raise
