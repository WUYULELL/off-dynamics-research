"""Deterministic nonlinear box-constrained least squares, in physical action units."""
import numpy as np
from scipy.optimize import lsq_linear


def solve(replay, action, wanted, scale, cfg, original):
    action = np.asarray(action, dtype=np.float64).copy()
    if not np.isfinite(action).all() or np.any(np.abs(action) > 1):
        raise ValueError('The recorded action must already lie in [-1, 1]')
    current = original
    residual = (current[0] - wanted) / scale
    cost = float(residual @ residual)
    initial_norm = np.sqrt(cost)
    calls = 1  # Includes the caller's replay of the original action.
    status = 'iteration_limit'
    iterations = 0
    while True:
        if np.max(np.abs(current[0] - wanted)) <= cfg['observation_noise_linf']:
            status = 'observation_noise_floor'
            break
        if iterations == cfg['max_outer_iterations']:
            break
        iterations += 1
        jac = np.empty((len(residual), len(action)), dtype=np.float64)
        for j, value in enumerate(action):
            trial = action.copy()
            step = cfg['finite_difference_absolute_step']
            trial[j] = value + step if value + step <= 1 else value - step
            jac[:, j] = (replay(trial)[0] - current[0]) / scale / (trial[j] - value)
            calls += 1
        gradient = jac.T @ residual
        projected = action - np.clip(action - gradient, -1, 1)
        if np.max(np.abs(projected)) <= cfg['gtol']:
            status = 'projected_gradient'
            break
        subproblem = lsq_linear(jac, -residual, bounds=(-1-action, 1-action),
                               method='bvls', tol=cfg['linear_tolerance'])
        if not np.isfinite(subproblem.x).all():
            raise FloatingPointError('Non-finite action update')
        if not subproblem.success:
            status = 'linear_subproblem_no_convergence'
            break
        if np.linalg.norm(subproblem.x) <= cfg['xtol']:
            status = 'step_tolerance'
            break
        for backtrack in range(cfg['backtracking_trials']):
            candidate = np.clip(action + (0.5 ** backtrack) * subproblem.x, -1, 1)
            evaluated = replay(candidate)
            calls += 1
            new_residual = (evaluated[0] - wanted) / scale
            new_cost = float(new_residual @ new_residual)
            if new_cost < cost:
                decrease = cost - new_cost
                action, current, residual, cost = candidate, evaluated, new_residual, new_cost
                if decrease <= cfg['ftol'] * (cost + decrease):
                    status = 'cost_tolerance'
                break
        else:
            status = 'no_decreasing_step'
        if status != 'iteration_limit':
            break
    return action, current, float(initial_norm), float(np.sqrt(cost)), iterations, calls, status
