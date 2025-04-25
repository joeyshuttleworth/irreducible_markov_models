import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
import argparse
import markov_builder as mb
from scipy.integrate import solve_ivp
from numba import njit
from markov_builder.example_models import construct_wang_chain
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

    ts = [0, protocol[-1, 0]]
    label_order = states
    state_labels, Q = mc.get_transition_matrix(label_order=label_order)

    steps_taken_vec = []

    tols = 10.0**np.array(list(range(-14, -2)))

    for tol in tols:
        for eliminated_state in states:
            labels = [s for s in sorted(states) if s not in eliminated_state]
            A, B = mc.eliminate_state_from_transition_matrix(labels,
                                                            use_parameters=True)

            state_labels = list(mc.graph)

            symbols = {}
            symbols['v'] = sp.sympify('V')
            symbols['p'] = sp.Matrix([sp.sympify(p) for p in parameter_labels])
            symbols['y'] = sp.Matrix([mc.get_state_symbol(s)
                                    for s in labels])

            GKr_index = len(parameter_labels) - 1

            default_parameters = np.array([val
                                        for key, val in mc.default_values.items()
                                        if (str(key) not in ['E_Kr', 'E_rev', 'V'] and val is not None)])

            mm = MarkovModel(symbols, A, B, mc.rate_expressions,
                            voltage=protocol_func,
                            default_parameters=default_parameters,
                            Q=Q, name=mc.name,
                            parameter_labels=parameter_labels,
                            GKr_index=GKr_index,
                            E_rev=-80.0)
            mm.protocol_description = protocol

            steps_taken = count_solver_steps(mm, protocol, ts, tol=tol)
            steps_taken_vec.append(steps_taken)

    steps_taken_vec = np.array(steps_taken_vec).reshape(len(tols), -1)
    print(steps_taken_vec)

def count_solver_steps(mm, protocol, ts, tol=1e-3):
    # start with equal proportion of channels in each state
    y0 = np.full(len(mm.get_state_labels()), 1.0)
    y0 = y0 / y0.shape[0]

    rhs_func = mm.get_rhs_func()

    def f_deriv(t, y, p):
        offset = p[-1]
        p = p[:-1].copy()
        v = mm.voltage(t, offset=offset)
        return rhs_func(y, p, v).flatten()

    # Add offste parameter
    p = np.append(mm.get_default_parameters(), 0.0)

    count = 0
    for step in protocol:
        tstart, tend, vstart, vend = step
        if tstart == tend or not np.isfinite(tend):
            continue
        tstart, tend = [0, ts[-1]]
        y = solve_ivp(f_deriv, [tstart, tend], y0, args=(p,), dense_output=True,
                        atol=tol, rtol=tol)
        y0 = y.y[:, -1].flatten()
        count += y.nfev

    return count

if __name__ == "__main__":
    main()
