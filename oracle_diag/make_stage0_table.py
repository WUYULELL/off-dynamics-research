"""Render stage-0 json results as markdown tables (results/stage0_tables.md)."""
import json
from common import ENVS, RESULTS


def fmt(x):
    return '{:.2e}'.format(x)


def row(name, d):
    s, r = d['state_l2_std'], d['reward_abs']
    return '| {} | {} | {} | {} | {} | {} | {} | {} |'.format(
        name, s['n'], fmt(s['mean']), fmt(s['median']), fmt(s['p95']), fmt(s['max']), fmt(r['mean']), fmt(r['p95']))


lines = []
for e in ENVS:
    d = json.load(open(RESULTS / 'stage0_{}.json'.format(e), encoding='utf-8'))
    t, s = d['target_check'], d['source_check']
    lines.append('### {}'.format(e))
    lines.append('')
    lines.append('frame_skip={}, dt={}, target buffer={} (of {}), qvel-clipped input: target {:.1%}, source {:.1%}'.format(
        d['frame_skip']['target'], d['dt']['target'], d['target_buffer_size'], d['target_full_size'],
        t['frac_qvel_clipped_input'], s['frac_qvel_clipped_input']))
    lines.append('recovery target: {} | source: {}'.format(t['recovery_status_counts'], s['recovery_status_counts']))
    lines.append('done agreement: target(obs) {:.4f}, target(recovered) {:.4f}, source {:.4f}, source(recovered) {:.4f}'.format(
        t['done_agreement_obs_replay'], t['done_agreement_recovered_replay'], s['done_agreement'], s['done_agreement_recovered']))
    lines.append('')
    lines.append('| check | n | state err mean | median | p95 | max | reward err mean | p95 |')
    lines.append('|---|---|---|---|---|---|---|---|')
    lines.append(row('T target->normal env, from obs, all', t['from_obs']['all']))
    lines.append(row('T   ... unclipped rows only', t['from_obs']['not_clipped']))
    if t['from_obs']['clipped'].get('state_l2_std', {}).get('n'):
        lines.append(row('T   ... clipped rows only', t['from_obs']['clipped']))
    lines.append(row('T* target->normal env, recovered state, all', t['from_recovered_state']['all']))
    lines.append(row('T*  ... excluding unrecoverable', t['from_recovered_state']['excluding_unrecoverable']))
    lines.append(row('S source->shifted env, from obs, all', s['vs_hdf5_next_observations']['all']))
    lines.append(row('S* source->shifted env, recovered state, all', s['recovered_vs_hdf5_next_observations']['all']))
    lines.append(row('X control: target rows->shifted env', t['control_target_rows_in_shifted_env']['all']))
    lines.append('')
lines.append('state err = ||s\'_replay - s\'_data|| in target-standardised units (state block of the seed-0 feature-bank std); reward err = |r_replay - r_data|.')
open(RESULTS / 'stage0_tables.md', 'w', encoding='utf-8').write('\n'.join(lines))
print('\n'.join(lines))
