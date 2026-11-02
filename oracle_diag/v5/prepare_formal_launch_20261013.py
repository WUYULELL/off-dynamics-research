"""Record the user's full formal authorization and verify frozen inputs on CPU."""
from common_v5 import *
import shutil


def main():
    launch = OUT / 'formal_launch1'
    assert not launch.exists(), 'Use a new launch directory; do not repeat this launch'
    assert not (OUT / 'gpu_formal_authorization.json').exists()
    plan = read(OUT / 'training_plan.json')
    protocol = read(OUT / 'protocol.json')
    jobs = [j for j in plan['jobs'] if j['kind'] == 'formal']
    assert len(jobs) == 30 and len({j['id'] for j in jobs}) == 30
    assert len({j['out'] for j in jobs}) == 30
    for stack, methods, seeds, steps in (
        ('A', A_METHODS, range(5), 500000),
        ('B', B_METHODS, range(1, 6), 1000000),
    ):
        selected = [j for j in jobs if j['stack'] == stack]
        assert [(j['method'], j['seed']) for j in selected] == [(m, s) for m in methods for s in seeds]
        assert all(j['steps'] == steps and j['phase'] == 'formal' for j in selected)
        assert all(Path(j['python']).is_file() and Path(j['entry']).is_file() for j in selected)
    assert all(not Path(j['out']).exists() for j in jobs), 'Formal outputs already exist'
    assert not (OUT / 'defer_B_until_A10.json').exists()
    delivered = read(OUT / 'smoke_delivery1/verified.json')
    assert delivered['passed'] and delivered['formal_jobs'] == 30
    assert read(OUT / 'smoke_audited.json')['passed']
    assert read(OUT / 'smoke_pipeline_status.json')['status'] == 'completed'
    for path, expected in delivered['input_sha256'].items():
        assert sha(path) == expected, 'Smoke-delivery input changed: ' + path
    smoke = [j for j in plan['jobs'] if j['kind'] == 'smoke']
    assert len(smoke) == 10
    for j in smoke:
        assert read(Path(j['out']) / 'audit.json')['passed']
        assert read(Path(j['out']) / 'completed.json')['steps'] == 5000
    protected = protected_check()
    assert protected == 3037
    launch.mkdir()
    for name, path in (
        ('active_execution_before.json', OUT / 'active_execution.json'),
        ('A_queue_status_before.json', OUT / 'queue_status.json'),
        ('B_queue_status_before.json', B_OUT / 'queue_status.json'),
    ):
        shutil.copy2(str(path), str(launch / name))
    authorization = dict(
        authorized=True, scope='formal', protocol_sha256=sha(OUT / 'protocol.json'),
        training_plan_sha256=sha(OUT / 'training_plan.json'),
        user_message='启动 A、B 全部 30 项正式训练，完成后继续更新桌面的跨域离线强化学习研究文档',
        recorded=time.time(), formal_jobs=30, A_jobs=15, B_jobs=15,
        scope_detail='Full frozen A and B formal queues, interim and final audited delivery, then update the Desktop research archive.',
        desktop_archive='C:/Users/TXin/Desktop/跨域离线强化学习研究',
    )
    write(OUT / 'gpu_formal_authorization.json', authorization)
    check_approval('formal')
    evidence = dict(
        passed=True, created=time.time(), protected_files=protected,
        smoke_jobs=10, formal_jobs=30, formal_outputs_absent=True,
        protocol_sha256=sha(OUT / 'protocol.json'),
        training_plan_sha256=sha(OUT / 'training_plan.json'),
        code_sha256=read(OUT / 'cpu_ready.json')['code_sha256'],
        smoke_delivery_inputs_rehashed=len(delivered['input_sha256']),
        authorization_sha256=sha(OUT / 'gpu_formal_authorization.json'),
        science_unchanged=True,
    )
    write(launch / 'prelaunch_verified.json', evidence)
    active = read(OUT / 'active_execution.json')
    active.update(
        status='formal_authorized_prelaunch_verified', authorized_scope='A and B full 30 formal jobs',
        formal_authorized=True, formal_started=False, authorization=str(OUT / 'gpu_formal_authorization.json'),
        launch=str(launch), A_launch=str(launch / 'A'), B_launch=str(launch / 'B'),
        queue_A_status=str(OUT / 'queue_status.json'), queue_B_status=str(B_OUT / 'queue_status.json'),
        formal_prelaunch=str(launch / 'prelaunch_verified.json'),
        desktop_archive=authorization['desktop_archive'], updated=time.time(),
        automation_id=None, automation_deleted=True,
    )
    write(OUT / 'active_execution.json', active, replace=True)
    print(json.dumps({k: evidence[k] for k in ('passed', 'protected_files', 'smoke_jobs', 'formal_jobs', 'science_unchanged')}))


if __name__ == '__main__':
    main()
