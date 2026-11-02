# -*- coding: utf-8 -*-
"""Preserve launch evidence and update the Desktop archive's current-status pages."""
from common_v5 import *
import datetime
import shutil


def link(path, label):
    return '[%s](<%s>)' % (label, Path(path).resolve().as_posix())


def main():
    launch = OUT / 'formal_launch1'
    archived = Path('C:/Users/TXin/Desktop/跨域离线强化学习研究')
    note = archived / '00_研究总览/v5正式训练状态_20261013.md'
    handoff = HERE / 'FORMAL_RUNNING_20261013.md'
    history = archived / '99_历史归档/旧版文档/20261013_正式训练启动前'
    evidence_copy = archived / '03_实验记录/03_自己的方法/Bellman相关验证/Oracle_v5/正式训练启动_20261013'
    assert not history.exists() and not evidence_copy.exists() and not note.exists() and not handoff.exists()
    now = datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat()
    jobs = [j for j in read(OUT / 'training_plan.json')['jobs'] if j['kind'] == 'formal']
    states = {}; progress = {}
    for stack, directory in [('A', OUT), ('B', B_OUT)]:
        s = read(directory / 'queue_status.json')
        assert s['status'] == 'running' and s['kind'] == s['phase'] == 'formal'
        j = next(j for j in jobs if j['id'] == s['current'])
        p = read(Path(j['out']) / 'progress.json')
        assert p['step'] >= 1000 and p['training_seconds'] > 0
        states[stack] = s; progress[stack] = p
    assert progress['A']['source_draws'] == progress['A']['target_draws'] == progress['A']['step'] * 64
    assert progress['B']['source_draws'] == 0 and progress['B']['target_draws'] == progress['B']['step'] * 128
    assert not (OUT / 'defer_B_until_A10.json').exists()
    evidence = dict(passed=True, checked=now, scope='Formal startup, not final results', states=states,
        progress=progress, science_unchanged=True, formal_jobs=30, completed_jobs=0,
        protocol_sha256=sha(OUT / 'protocol.json'), plan_sha256=sha(OUT / 'training_plan.json'),
        protected_prelaunch=read(launch / 'prelaunch_verified.json')['protected_files'],
        cuda_initialized=False)
    write(launch / 'startup_verified.json', evidence)
    active = read(OUT / 'active_execution.json')
    active.update(status='formal_running_startup_verified', automation_id='oracle-v5', automation_deleted=False,
        formal_started=True, formal_authorized=True, formal_handoff=str(handoff),
        formal_startup_verification=str(launch / 'startup_verified.json'),
        desktop_status=str(note), updated=time.time())
    write(OUT / 'active_execution.json', active, replace=True)
    message = read(OUT / 'gpu_formal_authorization.json')['user_message']
    handoff_text = f'''# Oracle v5 正式训练运行交接

更新：{now}。用户明确授权：“{message}”。完整 A/B 30 项已授权并启动，不再等待正式确认；旧 CPU_HANDOFF 和冒烟报告的待确认描述是历史状态。

## 实际执行

- 读取结果目录 active_execution.json 和 A/B queue_status.json，再核对实际进程命令行；PID 可能重用，不能只凭 PID 判断。
- 两个队列均为 run_queue_v5.py --stack A|B --kind formal --phase formal；队列内部逐组逐 seed 执行，A/B 同时运行。
- 2026-10-13 17:42（北京时间）启动。初始队列 PID：A=46752，B=17664；B 虚拟环境可能有 Python 启动器和实际训练子进程，不能误判重复队列。
- 计划：oracle_diag/results/v5_20261013/training_plan.json；冻结协议 protocol.json；正式授权 gpu_formal_authorization.json。
- A 日志 formal_launch1/A；B 日志 formal_launch1/B（均在 A 结果目录）；队列 stderr/stdout 在 formal_launch1 根目录。
- A 输出 oracle_diag/results/v5_20261013/f1；B 输出 reproductions/otdf_v5/results_20261013/f1。
- 启动前复核10项冒烟、全部3037项旧文件、冻结18脚本与冒烟交付输入哈希均通过；证据 formal_launch1/prelaunch_verified.json、startup_verified.json。
- 初次启动检查：A CRITIC-ORACLE seed0 已到 {progress['A']['step']} 步，B IQL-TARGET seed1 已到 {progress['B']['step']} 步。这是快照，以最新 progress 为准。

## 冻结范围和后续交付

A：CRITIC-ORACLE → CRITIC-SOURCE → Q-ORACLE，每组 seeds0–4，各500k，64+64抽样，source系数1.0。B：IQL-TARGET → IQL-TARGET+REG → OTDF-NOREG，每组seeds1–5，各1M；启用REG另VAE预训练10k。保持完整30项，不自动缩种子，不更改损失、预算或判据。不得改旧代码和结果，不得重跑冒烟或历史基线。

每项GPU结束后由 audit_v5.py 在CPU审计；审计失败先修审计，不应重跑已完成训练。真实训练故障保留失败目录，从相同seed第0步在新attempt恢复，不拼接、不按回报挑选。新代码修改须记录前后哈希和依据。

progress_v5.py 生成 interim_ready/final_ready；首组A/B各五seed全部审计完成后，运行 report_v5.py --stage interim，人工独立核对QA1/QA2/QB1并交付中期报告，记录交付凭据，队列继续。正式吞吐低于单独一半时产生 defer_B_until_A10.json，在B作业边界等待A前10项，保留当前B训练并通知用户。

最终运行 report_v5.py --stage final，并独立核对原始TensorBoard：A460k–500k、B960k–1M Last-5；逐seed差、样本SD、SE、正号数和Final。补充两项描述性比较、分组均值及对QA/QB的清晰解释。核验实际shifted评估、归一化、抽样、初始化、模型有限性、source/oracle/OT行对齐及3037保护哈希。脚本拒绝覆盖已存在报告，重审时须保留旧版本并写新attempt证据。

2SE且至少4个正号为工程门槛，不能声称严格95%置信结论、数学上界、等效或所有source有害。A/B分数不可横排，A5仅描述性；QA2只检验移除直接source actor损失，source仍可通过Q/V间接影响actor；QB2包含batch变化，VAE会改变后续RNG。保留单任务、target约8条轨迹段、V为学得函数、oracle恢复隐藏状态等解释范围。

## 桌面资料库维护

目标：C:/Users/TXin/Desktop/跨域离线强化学习研究。本轮只更新入口、研究思路、当前边界与启动状态，并备份更新前页面到99_历史归档/旧版文档/20261013_正式训练启动前。研究进度总表.xlsx和已复制的冒烟/CPU报告当前仍是归档时快照。

完成后继续更新00总览中的研究进度总表.xlsx、研究问题与当前思路.md、文件位置索引.md，按03实验记录、04结果分析、05汇报材料放置新报告与轻量证据，更新清单和哈希记录。不要搬动原始数据模型，不要覆盖用户新增修改。保留报告旧版和更新前文档。表格更新应用spreadsheets技能；参考 outputs/research_archive_20261013 下构建/验证脚本，但不要直接重跑拒绝已有根目录的 build_archive.py。当前归档核验记录只证明最初归档快照，之后更新另有增量核验凭据。

自动跟进ID oracle-v5，每小时检查；正常推进保持安静，中期、完成、故障或需要用户操作时通知。交付最终报告与桌面资料链接并完成更新后删除自动跟进。
'''
    readme = archived / 'README_从这里开始.md'
    overview = archived / '00_研究总览/研究问题与当前思路.md'
    boundary = archived / '02_研究方案/当前方案/当前执行边界.md'
    location = archived / '00_研究总览/文件位置索引.md'
    edits = {}
    replacements = {
        readme: [('当前：v4已完成；v5冒烟已完成，正式A/B共30项待确认。本次仅归档，没有启动训练。',
                  '当前：v4已完成；用户已授权并启动v5正式A/B全部30项。' + link(note, '正式训练启动状态') + '。进度总表和归档报告保留归档时快照，正式训练完成后统一更新。')],
        overview: [('更新：2026-10-13。这是对现有材料的归档综述；本次没有新增训练，也没有重新执行所有历史实验审计。',
                    '更新：2026-10-13。本文的历史结果来自归档材料；最新变更为用户已授权并启动v5正式A/B全部30项，尚无最终结果。'),
                   ('**CPU检查、A5动作诊断、10项GPU冒烟、历史复用和并行吞吐测试已完成。正式A/B共30项尚未授权、未启动。**',
                    '**CPU检查、A5动作诊断、10项GPU冒烟、历史复用和并行吞吐测试已完成。正式A/B共30项已获授权，于2026-10-13 17:42并行启动。** ' + link(note, '正式训练启动状态')),
                   ('- 正式确认后，先交付QA1、QA2、QB1的中期结果，再继续已授权的剩余比较。',
                    '- 两边首组各5个seed完成并通过核验后，交付QA1、QA2、QB1中期结果，队列继续；全部完成后更新桌面进度表、实验记录和结果分析。')],
        boundary: [('待完成：用户确认后的30项正式训练、中期和最终统计。',
                    '进行中：用户已确认的完整30项正式训练。待完成：中期和最终统计、报告及桌面资料更新。'),
                   ('本次“归档”请求没有授权启动正式训练。A/B数据、损失、种子、预算与判据保持冻结。',
                    '2026-10-13用户随后明确授权“启动 A、B 全部 30 项正式训练，完成后继续更新桌面的跨域离线强化学习研究文档”。两条队列已启动，A/B数据、损失、种子、预算与判据保持冻结。' + link(note, '启动状态与授权凭据'))],
        location: [('## 优先阅读', '## 最新执行状态\n\n' + link(note, 'v5全部30项正式训练已启动（2026-10-13）') + '。原有归档清单仍记录最初复制快照，启动增量及旧版备份另附核验记录；完整表格与结果材料将在正式训练结束后更新。\n\n## 优先阅读')],
    }
    for path, changes in replacements.items():
        text = path.read_text(encoding='utf-8')
        for old, new in changes:
            assert text.count(old) == 1, (str(path), old)
            text = text.replace(old, new)
        edits[path] = text
    history.mkdir(parents=True); evidence_copy.mkdir(parents=True)
    changes = []
    for path, text in edits.items():
        backup = history / path.relative_to(archived)
        backup.parent.mkdir(parents=True, exist_ok=True)
        before = sha(path); shutil.copy2(str(path), str(backup))
        assert sha(backup) == before
        path.write_text(text, encoding='utf-8')
        changes.append(dict(path=str(path), backup=str(backup), before_sha256=before, after_sha256=sha(path)))
    with open(handoff, 'x', encoding='utf-8') as f:
        f.write(handoff_text)
    copied = []
    for source in [OUT/'gpu_formal_authorization.json', launch/'prelaunch_verified.json', launch/'launch_processes.json', launch/'startup_verified.json', handoff]:
        target = evidence_copy/source.name
        assert not target.exists()
        shutil.copy2(str(source), str(target)); assert sha(source) == sha(target)
        copied.append(dict(source=str(source), archived=str(target), sha256=sha(target)))
    text = f'''# v5 正式训练启动状态

更新时间：{now}。用户已明确授权 A、B 全部30项正式训练，并要求完成后更新本资料库。

**2026-10-13 17:42，两条队列已并行启动。当前为运行阶段，还没有本轮最终结论。**

| 队列 | 顺序 | 种子 | 每项策略步数 | 项数 |
|---|---|---|---:|---:|
| A 路径归因 | CRITIC-ORACLE → CRITIC-SOURCE → Q-ORACLE | 0–4 | 500000 | 15 |
| B OTDF对照 | IQL-TARGET → IQL-TARGET+REG → OTDF-NOREG | 1–5 | 1000000 | 15 |

启动复核时，A首项已到{progress['A']['step']}步、B首项已到{progress['B']['step']}步；这是一份时间快照，实时状态见下方源文件。A的source/target抽样64+64，B首项source抽样为0，计数正确。预检10项冒烟、冻结脚本和3037项旧文件哈希全部通过。

根据此前冒烟预计总排期约3–4天，实际受正式训练吞吐影响。两边首组各5个seed完成审计后先交付中期报告，训练继续；全部完成后独立审计并更新本资料库的进度总表、研究思路、文件索引、实验记录、结果分析和报告。

进度总表.xlsx与复制的旧冒烟报告当前仍保留归档时快照。当前授权与执行状态以本页及实时状态文件为准；历史报告中的“待确认”不代表仍在等待。

- {link(evidence_copy/'gpu_formal_authorization.json', '正式授权原文与协议哈希')}
- {link(evidence_copy/'startup_verified.json', '启动核验快照')}
- {link(evidence_copy/'FORMAL_RUNNING_20261013.md', '执行与交付说明')}
- {link(OUT/'active_execution.json', '最新执行入口')}
- {link(OUT/'queue_status.json', 'A队列实时状态')}
- {link(B_OUT/'queue_status.json', 'B队列实时状态')}
- {link(history, '本次更新前的桌面文档备份')}

自动跟进已启用。正常运行不重复通知；中期结果、全部完成、故障或需要处理的变化时通知。A/B协议不同，未来结果分别比较，不将分数横向排名。
'''
    with open(note, 'x', encoding='utf-8') as f:
        f.write(text)
    write(archived/'00_研究总览/正式训练启动更新核验_20261013.json',
        dict(passed=True, created=now, scope='Incremental startup update; original archive manifest preserved',
            changed=changes, copied=copied, note=dict(path=str(note),sha256=sha(note)),
            workbook_updated=False, workbook_note='Initial archive snapshot; update after audited results'))
    print(json.dumps(dict(passed=True, current_pages_updated=len(changes), startup_copies=len(copied),
        A_step=progress['A']['step'], B_step=progress['B']['step'], handoff=str(handoff)), ensure_ascii=False))


if __name__ == '__main__':
    main()
