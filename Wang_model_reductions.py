import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import argparse
import markov_builder as mb
import scipy
import cma

from scipy.integrate import solve_ivp
from numba import njit
from markov_builder.example_models import construct_wang_chain
from markov_builder.example_models import construct_mazhari_chain
from markovmodels import MarkovModel
import myokit as mk

def main():
    arg_parser = argparse.ArgumentParser()

    # Setup protocol
    mk_protocol = mk.load_protocol('simplified-staircase.mmt')
    protocol = []
    # for event in mk_protocol.events():
    #     duration = event.duration()
    #     end_t = mk_protocol.characteristic_time()
    #     start_t = end_t - duration
    #     level = event.level()
    #     protocol.append((start_t, end_t, level, level))

    # protocol = np.vstack(protocol).astype(np.float64)
    # protocol[:-1, 1] = protocol[1:, 0].copy()
    # print(protocol)

    protocol = np.array([[0, 1000.0, -80.0, -80.0],
                         [1000.0, 2000.0, 40.0, 40.0],
                         [2000.0, 3000.0, 0.0, 0.0],
                         [3000.0, 4000.0, -80.0, -80.0]])

    holding_potential = -80.0

    @njit
    def protocol_func(t, offset=0, protocol_description=protocol):
        t = t + offset
        if t < 0 or t >= protocol[-1][1]:
            return holding_potential

        for i in range(len(protocol)):
            if t <= protocol[i][1]:
                if np.abs(protocol[i][3] - protocol[i][2]) > 0.0:
                    return protocol[i][2] + (t - protocol[i][0])*(protocol[i][3]-protocol[i][2])/(protocol[i][1] - protocol[i][0])
                else:
                    return protocol[i][3]

    global args
    args = arg_parser.parse_args()
    mc = construct_wang_chain()
    states = mc.get_states()

    parameter_labels = [key
                        for key, val in mc.default_values.items()
                        if str(key) not in ['E_Kr', 'E_rev', 'V'] and val is not None]

    label_order = states
    state_labels, Q = mc.get_transition_matrix(label_order=label_order)

    steps_taken_vec = []

    tols = 10.0**np.array(list(range(-12, -2)))

    tend = protocol[-1, 1]
    ts = np.linspace(0, int(tend), int(tend/100))

    ref_res = get_reference_solution(mc, protocol, ts, protocol_func)

    plt.plot(ts, ref_res)
    plt.savefig("reference_sol")

    eliminated_state = states[0]
    labels = [s for s in states if s not in eliminated_state]
    A, B = mc.eliminate_state_from_transition_matrix(labels,
                                                     use_parameters=True)

    GKr_index = len(parameter_labels) - 1
    default_parameters = np.array([val
                                for key, val in mc.default_values.items()
                                if (str(key) not in ['E_Kr', 'E_rev', 'V'] and val is not None)])

    symbols = {}
    symbols['v'] = sp.sympify('V')
    symbols['p'] = sp.Matrix([sp.sympify(p) for p in parameter_labels])
    symbols['y'] = sp.Matrix([mc.get_state_symbol(s)
                                for s in labels])
    mm = MarkovModel(symbols, A, B, mc.rate_expressions,
                     voltage=protocol_func,
                     default_parameters=default_parameters,
                     Q=Q, name=mc.name,
                     parameter_labels=parameter_labels,
                     GKr_index=GKr_index,
                     E_rev=-80.0)
    rates_func = mm.get_rates_func()

    Q_func = njit(sp.lambdify((mm.rates_dict.keys(), mm.v), Q))

    @njit
    def _Q_func(v):
        rates = rates_func(default_parameters, v).flatten()
        return Q_func(rates, v)

    voltages = np.unique(protocol[2, :])

    x0 = np.full(Q.shape[0], 1.0)
    x0 = np.random.normal(0, 1, Q.shape[0])
    x0 = x0 / np.linalg.norm(x0)

    x0 = np.zeros(Q.shape[0])
    x0[-1] = 1

    # initial_score = np.nan
    # while not np.isfinite(initial_score):
    #     x0 = np.full(Q.shape[0], 1.0)
    #     x0 = np.random.normal(0, 1, Q.shape[0])
    #     x0 = x0 / np.linalg.norm(x0)

    initial_score = opt_func(x0, _Q_func, voltages)

    # Run the optimization
    sigma = 20.0
    es = cma.CMAEvolutionStrategy(x0, sigma, {'maxiter': 1_000_000,
                                              'popsize': 20})

    _opt_func = lambda x: opt_func(x, _Q_func, voltages)
    es.optimize(_opt_func)
    print(es.stop())

    resvec = es.result.xbest
    resvec /= np.linalg.norm(resvec)
    print(resvec)

    C, d = general_transform_with_reduction_vec(Q, resvec)
    mm = MarkovModel(symbols, C, d, mc.rate_expressions,
                     voltage=protocol_func,
                     default_parameters=default_parameters,
                     Q=Q, name=mc.name,
                     parameter_labels=parameter_labels,
                     GKr_index=GKr_index,
                     E_rev=-80.0)

    ortho_basis = construct_orthonormal_basis(resvec)
    N = Q.shape[0]
    T = np.vstack((ortho_basis[:-1, :].astype(np.float64), np.full((1, N), 1.0)))
    print(T, np.linalg.det(T), np.linalg.cond(T))

    y0 = np.full(len(mm.get_state_labels()) + 1, 1.0)
    y0 = y0 / (len(y0))
    y0 = (T @ y0).flatten()[:-1]
    count, res = count_solver_steps(mm, protocol, ts, tol=1e-6, y0=y0)

    res = np.hstack([res, np.full((res.shape[0], 1), 1.0)])
    res = np.linalg.solve(T, res.T).T

    rmse = np.sqrt(np.mean((res - ref_res)**2))
    plt.clf()
    plt.plot(ts, res-ref_res)
    plt.yscale('log')
    plt.savefig("log_error_plot")
    print(rmse)
    print(count)

    rmses = []
    for tol in tols:
        for i, eliminated_state in enumerate(states):
            labels = [s for s in states if s not in eliminated_state]
            A, B = mc.eliminate_state_from_transition_matrix(labels,
                                                            use_parameters=True)

            GKr_index = len(parameter_labels) - 1
            default_parameters = np.array([val
                                        for key, val in mc.default_values.items()
                                        if (str(key) not in ['E_Kr', 'E_rev', 'V'] and val is not None)])

            symbols = {}
            symbols['v'] = sp.sympify('V')
            symbols['p'] = sp.Matrix([sp.sympify(p) for p in parameter_labels])
            symbols['y'] = sp.Matrix([mc.get_state_symbol(s)
                                      for s in labels])
            mm = MarkovModel(symbols, A, B, mc.rate_expressions,
                            voltage=protocol_func,
                            default_parameters=default_parameters,
                            Q=Q, name=mc.name,
                            parameter_labels=parameter_labels,
                            GKr_index=GKr_index,
                            E_rev=-80.0)
            mm.protocol_description = protocol

            steps_taken, res = count_solver_steps(mm, protocol, ts, tol=tol)
            steps_taken_vec.append(steps_taken)

            missing_state_res = 1.0 - res.sum(axis=1).flatten()
            res = np.insert(res, i,
                            missing_state_res, axis=1)
            rmse = np.sqrt(np.mean((res - ref_res)**2))
            rmses.append(rmse)

    steps_taken_vec = np.array(steps_taken_vec).reshape(len(tols), -1)
    rmses = np.array(rmses).reshape(len(tols), -1)

    print(steps_taken_vec, rmses)


def get_reference_solution(mc, protocol, ts, voltage_func, tol=1e-13):
    # start with equal proportion of channels in each state
    state_labels, Q = mc.get_transition_matrix()
    y0 = np.full(Q.shape[0], 1.0)
    y0 = y0 / float(len(y0))

    y = sp.Matrix([mc.get_state_symbol(s)
                   for s in state_labels])
    parameter_labels = [key
                        for key, val in mc.default_values.items()
                        if str(key) not in ['E_Kr', 'E_rev', 'V'] and val is not None]
    v = sp.sympify('V')
    inputs = (y, parameter_labels, v)

    rhs_expr = (Q.T @ y).subs(mc.rate_expressions)

    rhs_func = njit(sp.lambdify(inputs, rhs_expr))

    def f_deriv(t, y, p):
        offset = p[-1]
        p = p[:-1].copy()
        v = voltage_func(t, offset=offset)
        return rhs_func(y.flatten(), p.flatten(), np.float64(v)).flatten()

    # Add offset parameter
    default_parameters = np.array([val
                                   for key, val in mc.default_values.items()
                                   if (str(key) not in ['E_Kr', 'E_rev', 'V'] and val is not None)])

    p = np.append(default_parameters, 0.0)
    res = []
    for step in protocol:
        tstart, tend, vstart, vend = step
        if tstart == tend:
            continue

        _ts = ts[(ts >= tstart) & (ts <= tend)]

        if tend not in _ts:
            _ts = np.append(_ts, tend)

        y = solve_ivp(f_deriv, (tstart, tend), y0, args=(p,), dense_output=False,
                        atol=tol, rtol=tol, method='RK45', t_eval=_ts)
        y0 = y.y[:, -1].flatten()
        res.append(y.y[:, :-1])

    res.append(y0[:, None])
    res = np.hstack(res).T
    res = res / res.sum(axis=1)[:, None]
    return res


def count_solver_steps(mm, protocol, ts, tol=1e-3, y0=None):
    # start with equal proportion of channels in each state

    if y0 is None:
        y0 = np.full(len(mm.get_state_labels()), 1.0)
        y0 = y0 / (len(y0) + 1.0)

    rhs_func = mm.get_rhs_func()

    def f_deriv(t, y, p):
        offset = p[-1]
        p = p[:-1].copy()
        v = mm.voltage(t, offset=offset)
        return rhs_func(y, p, v).flatten()

    # Add offset parameter
    p = np.append(mm.get_default_parameters(), 0.0)

    jacobian = sp.Matrix(mm.A)
    jac_func = njit(sp.lambdify(('t', 'y', mm.p), jacobian))

    count = 0
    res = []
    for step in protocol:
        tstart, tend, vstart, vend = step
        if tstart == tend:
            continue
        _ts = ts[(ts >= tstart) & (ts <= tend)]

        if tend not in _ts:
            _ts = np.append(_ts, tend)

        y = solve_ivp(f_deriv, (tstart, tend), y0, args=(p,), atol=tol,
                      rtol=tol, method='RK45', dense_output=True)
        # jac=jac_func)
        y0 = y.y[:, -1].flatten()
        count += y.nfev
        res.append(y.sol(_ts[:-1]))

    res.append(y.sol(np.array([_ts[-1]])))
    res = np.hstack(res).T
    return count, res


def general_transform_with_reduction_vec(Q, e1):
    N = Q.shape[0]

    ortho_basis = construct_orthonormal_basis(e1)
    T = np.vstack((ortho_basis[:-1, :].astype(np.float64), np.full((1, N), 1.0)))

    if np.linalg.det(T) < 1e-15:
        return np.full((N, N), np.nan), np.full(N, np.nan)

    W = T @ Q.T @ np.linalg.inv(T)
    C = W[:-1, :-1]
    d = 1 * W[:-1, -1]

    return C, d


def opt_func(x, Q_func, voltages):
    x = x / np.sqrt(np.sum(x**2))

    score = 0

    conds = np.empty(len(voltages))

    for i, v in enumerate(voltages):
        Q = Q_func(v).astype(np.float64)
        C, d = general_transform_with_reduction_vec(Q, x)

        if not np.all(np.isfinite(C)):
            return np.inf

        conds[i] = np.linalg.cond(C, p=np.inf)

    return np.log10(np.max(conds))


def construct_orthonormal_basis(v1):
    v1 = v1 / np.linalg.norm(v1)
    n = v1.shape[0]

    basis = np.full((n, n), 0.0)
    basis[0, :] = v1

    ones = np.eye(n)

    c = 1
    for i in range(n):
        if c >= n:
            break

        v = ones[i, :].flatten()
        for j in range(c):
            v -= np.dot(v, basis[j, :]) * basis[j, :]

        if np.linalg.norm(v) > 1e-18:
            v /= np.linalg.norm(v)
            basis[c, :] = v
            c += 1

    ret_val = basis.copy()
    ret_val[:-1, :] = basis[1:, :]
    ret_val[-1, :] = basis[0, :]
    return ret_val


if __name__ == "__main__":
    main()
