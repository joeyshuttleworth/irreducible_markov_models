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


holding_potential = -80.0
scaling_factor = 1.0

def main():
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--model", default='Wang')

    # Setup protocol
    mk_protocol = mk.load_protocol('simplified-staircase.mmt')
    protocol = []
    t_cur = 0
    for event in mk_protocol.events():
        duration = event.duration()
        start_t = t_cur
        end_t = start_t + duration
        t_cur = end_t
        level = event.level()
        protocol.append((start_t, end_t, level, level))

    protocol = np.vstack(protocol).astype(np.float64)
    print(protocol)

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
    if args.model == 'Wang':
        mc = construct_wang_chain()
    elif args.model == 'Mazhari':
        mc = construct_mazhari_chain()
    elif args.model[:5] == 'model':
        model_no = int(args.model[5:])
        from markov_builder.models.thirty_models import (
            model_00,
            model_01,
            model_02,
            model_03,
            model_04,
            model_05,
            model_06,
            model_07,
            model_08,
            model_09,
            model_10,
            model_11,
            model_12,
            model_13,
            model_14,
            model_20,
            model_30,
        )

        # Missing models here
        model_15 = None
        model_16 = None
        model_17 = None
        model_18 = None
        model_19 = None
        model_20 = None
        model_21 = None
        model_22 = None
        model_23 = None
        model_24 = None
        model_25 = None
        model_26 = None
        model_27 = None
        model_28 = None
        model_29 = None

        thirty_models = [
            model_00, model_01, model_02, model_03, model_04,
            model_05, model_06, model_07, model_08, model_09, model_10,
            model_11, model_12, model_13, model_14, model_15, model_16, model_17,
            model_18, model_19, model_20, model_21, model_22, model_23, model_24, model_25,
            model_26, model_27, model_28, model_29, model_30
        ]
        mc = thirty_models[model_no]()

        if not mc.is_connected():
            raise NotImplementedError

    states = mc.get_states()

    parameter_labels = [key
                        for key, val in mc.default_values.items()
                        if str(key) not in ['E_Kr', 'E_rev', 'V'] and val is not None]

    label_order = states
    state_labels, Q = mc.get_transition_matrix(label_order=label_order)

    tols = 10.0**np.array(list(range(-8, -2)))

    tend = np.array(protocol)[-1, 1]
    ts = np.linspace(0, int(tend), int(tend) + 1)

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

    voltages = np.unique(protocol[:, 2:])

    x0 = np.full(Q.shape[0], 1.0)

    initial_score = np.nan

    c = 0
    while not np.isfinite(initial_score):
        x0 = np.full(Q.shape[0], 1.0)
        x0 = np.random.normal(0, 1, Q.shape[0])
        x0 = x0 / np.linalg.norm(x0, 2)
        initial_score = opt_func(x0, _Q_func, voltages)

    # Run the optimization
    sigma = 1.0
    es = cma.CMAEvolutionStrategy(x0, sigma, {'maxiter': 1_000_000,
                                              'tolstagnation': 5000,
                                              'tolfacupx': 1_000_000_000,
                                              'popsize': 20,
                                              'tolflatfitness': 100})

    _opt_func = lambda x: opt_func(x, _Q_func, voltages)

    # This is a stochstic optimiser so the results are random.
    # There's a chance that it fails to find a good vector
    # [-0.01116938 -0.01036687 -0.71767416 -0.01100285  0.69612535]
    # is an example of a vector which works well
    es.optimize(_opt_func)
    print(es.stop())

    resvec = es.result.xbest
    resvec /= np.linalg.norm(resvec, 2)
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
    T = ortho_basis.copy()

    print(T, np.linalg.det(T), np.linalg.cond(T))

    y0 = np.full(len(mm.get_state_labels()) + 1, 1.0)
    y0 = y0 / (len(y0))
    y0 = (T @ y0).flatten()[:-1]

    rmse_vec = []
    steps_taken_vec = []
    for tol in tols:
        count, ivp_res = count_solver_steps(mm, protocol, ts, tol=tol, y0=y0)
        steps_taken_vec.append(count)
        ivp_res = np.hstack([ivp_res, np.full((ivp_res.shape[0], 1), 1.0/np.sqrt(N))])
        ivp_res = np.linalg.solve(T, ivp_res.T).T
        rmse = np.sqrt(np.mean((ivp_res - ref_res)**2))
        rmse_vec.append(rmse)
        plt.clf()
        plt.plot(ts, ivp_res-ref_res)
        plt.yscale('log')
        plt.savefig("log_error_plot")

    rmse_vec1 = np.array(rmse_vec.copy())
    steps_taken_vec1 = np.array(steps_taken_vec)

    rmses = []
    steps_taken_vec = []
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
            plt.clf()
            plt.plot(ts, res)
            plt.savefig(f"{eliminated_state}_{tol:1e}_sol.pdf")

    rmses_full = []
    steps_taken_vec_full = []

    for tol in tols:
        labels = states
        A, B = mc.eliminate_state_from_transition_matrix(labels[:-1],
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

        steps_taken, res = count_solver_steps(mm, protocol, ts, tol=tol, use_Q=True)
        steps_taken_vec_full.append(steps_taken)

        rmse = np.sqrt(np.mean((res - ref_res)**2))
        rmses_full.append(rmse)

    steps_taken_vec = np.array(steps_taken_vec).reshape(len(tols), -1)
    rmses = np.array(rmses).reshape(len(tols), -1)

    rmses = np.hstack([rmses, rmse_vec1[:, None], np.array(rmses_full)[:, None]])
    steps_taken_vec = np.hstack([steps_taken_vec, steps_taken_vec1[:, None],
                                 np.array(steps_taken_vec_full)[:, None]])

    print(rmses, steps_taken_vec)
    plt.clf()
    plt.plot(steps_taken_vec, rmses, label=[s for s in state_labels] + ['optimised', 'full'])
    plt.yscale('log')
    # plt.xscale('log')
    plt.legend()

    plt.savefig("reduction_comparison")

def get_reference_solution(mc, protocol, ts, voltage_func, tol=1e-10):
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
    Q_func = njit(sp.lambdify(inputs, Q.T.subs(mc.rate_expressions)))

    def jac_func(t, y, p):
        offset = p[-1]
        p = p[:-1].copy()
        v = voltage_func(t, offset=offset)
        return Q_func(y, p, v)

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

        y = solve_ivp(f_deriv, (tstart, tend), y0, args=(p,), dense_output=True,
                      atol=tol, rtol=tol, method='BDF', jac=jac_func)
        _res = y.sol(_ts)
        y0 = _res[:, -1].flatten()
        res.append(_res[:, :-1])

    res.append(y0[:, None])
    res = np.hstack(res).T
    res = res / res.sum(axis=1)[:, None]
    return res


def count_solver_steps(mm, protocol, ts, tol=1e-3, y0=None, use_Q=False):
    # start with equal proportion of channels in each state

    inputs = (mm.y, mm.p, mm.v)
    A_mat = mm.A.subs(mm.rates_dict)
    A_func = njit(sp.lambdify(inputs, mm.A.subs(mm.rates_dict)))

    Q_mat = mm.Q.subs(mm.rates_dict)
    Q_func = njit(sp.lambdify(inputs, mm.Q.subs(mm.rates_dict)))

    if y0 is None:
        y0 = np.full(len(mm.y), 1.0) / Q_mat.shape[0]

    if use_Q:
        rhs_func = njit(sp.lambdify(inputs, (mm.Q.T @ mm.y).subs(mm.rates_dict)))
    else:
        rhs_func = mm.get_rhs_func()

    voltage_func = mm.voltage
    def f_deriv(t, y, p):
        offset = p[-1]
        p = p[:-1].copy()
        v = voltage_func(t, offset=offset)
        return rhs_func(y, p, v).flatten()

    def jac_func(t, y, p=mm.default_parameters):
        offset = p[-1]
        p = p[:-1].copy()
        v = voltage_func(t, offset=offset)
        if use_Q:
            return Q_func(y, p, v)
        else:
            return A_func(y, p, v)

    # Add offset parameter
    p = np.append(mm.get_default_parameters(), 0.0)

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
                      rtol=tol, method='BDF', dense_output=True, jac=jac_func)

        y0 = y.y[:, -1].flatten()
        count += y.nfev
        res.append(y.sol(_ts[:-1]))

    res.append(y.sol(np.array([_ts[-1]])))
    res = np.hstack(res).T
    return count, res


def general_transform_with_reduction_vec(Q, e1):
    N = Q.shape[0]

    ortho_basis = construct_orthonormal_basis(e1)
    T = ortho_basis.copy()

    W = T @ Q.T @ np.linalg.inv(T)
    C = W[:-1, :-1]
    d = (1/np.sqrt(N)) * W[:-1, -1]

    return C, d


def opt_func(x, Q_func, voltages):
    x = x / np.sqrt(np.sum(x**2))

    score = 0

    conds = np.empty(len(voltages))

    T = construct_orthonormal_basis(x)
    T[-1, :] = 1 / np.sqrt(len(x))

    norm_type = 2
    for i, v in enumerate(voltages):
        Q = Q_func(v).astype(np.float64)

        W = T @ Q.T @ np.linalg.inv(T)
        C = W[:-1, :-1]

        if not np.all(np.isfinite(C)):
            return np.inf

        conds[i] = np.linalg.cond(C, p=norm_type)

    return np.log10(np.mean(conds))


def construct_orthonormal_basis(v1):
    v1 = v1 / np.linalg.norm(v1, 2)
    n = v1.shape[0]

    basis = np.full((n, n), 0.0)
    basis[0, :] = np.full(n, 1/np.sqrt(n))
    # basis[0, :] = 0.0
    basis[1, :] = v1

    ones = np.eye(n)

    c = 2
    for i in range(n):
        if c >= n:
            break

        v = ones[i, :].flatten()
        for j in range(c):
            v -= np.dot(v, basis[j, :]) * basis[j, :]

        if np.linalg.norm(v, 2) > 1e-10:
            v /= np.linalg.norm(v, 2)
            basis[c, :] = v
            c += 1

    ret_val = basis.copy()
    ret_val[:-1, :] = basis[1:, :]
    ret_val[-1, :] = 1/np.sqrt(n)
    return ret_val


if __name__ == "__main__":
    main()
