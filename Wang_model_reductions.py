import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import argparse
import markov_builder as mb
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
    for event in mk_protocol.events():
        duration = event.duration()
        end_t = mk_protocol.characteristic_time()
        start_t = end_t - duration
        level = event.level()
        protocol.append((start_t, end_t, level, level))

    protocol = np.vstack(protocol).astype(np.float64)

    protocol = np.array([[0, 1000.0, -80.0, -80.0],
                         [1000.0, 2000.0, 40.0, 40.0],
                         # [2000.0, 3000.0, 0.0, 0.0],
                         [2000.0, 3000.0, -80.0, -80.0]])

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

    tols = 10.0**np.array(list(range(-8, -2)))

    tend = protocol[-1, 1]
    ts = np.linspace(0, tend, int(tend/10))

    ref_res = get_reference_solution(mc, protocol, ts, protocol_func)

    plt.plot(ts, ref_res)
    plt.savefig("reference_sol")

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

    rates_func = mm.get_rates_func()

    Q_func = njit(sp.lamdfiy((mm.rates_dict.keys(), mm.v), Q))
    @njit
    def _Q_func(v):
        rates = rates_func(default_p, v)
        return Q_func(rates, v)

    scipy.optimize.minimize(opt_func, x0, args=(Q_func, voltages))


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

        _ts = ts[(ts >= tstart) & (ts < tend)]
        y = solve_ivp(f_deriv, (_ts.min(), _ts.max()), y0, args=(p,), dense_output=False,
                        atol=tol, rtol=tol, method='RK45', t_eval=_ts)
        y0 = y.y[:, -1].flatten()
        res.append(y.y[:, :])

    res.append(y.y[:, -1][:, None])
    res = np.hstack(res).T
    res = res / res.sum(axis=1)[:, None]
    return res


def count_solver_steps(mm, protocol, ts, tol=1e-3):
    # start with equal proportion of channels in each state
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
        _ts = ts[(ts >= tstart) & (ts < tend)]
        y = solve_ivp(f_deriv, (_ts.min(), _ts.max()), y0, args=(p,), dense_output=False,
                        atol=tol, rtol=tol, method='RK45', t_eval=_ts)
        # jac=jac_func)
        y0 = y.y[:, -1].flatten()
        count += y.nfev
        res.append(y.y[:, :])

    res.append(y.y[:, -1][:, None])
    res = np.hstack(res).T
    return count, res


def general_transform_with_reduction_vec(Q, e1):
    N = Q.shape[0]

    vecs = np.vstack([e1, np.ones(N, N)])
    ortho_basis = np.qr(vecs)

    T = sp.Matrix([ortho_basis[:-1, :], sp.ones(1, N)])

    W = T @ Q.T @ T**(-1)
    C = W[:-1, :-1]
    d = 1 * W[:, -1]

    return C, d


@njit
def opt_func(x, Q_func, voltages):
    x = x / np.sqrt(np.sum(x**2))

    cond = 0
    for v in voltages:
        Q = Q_func(v)
        C, d = general_transform_with_reduction_vec(Q, x)

        cond = np.linalg.norm(mat, ord=2)
        score += cond

    return score


if __name__ == "__main__":
    main()
