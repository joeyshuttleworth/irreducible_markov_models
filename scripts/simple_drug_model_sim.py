import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import markov_builder
import matplotlib
import cma
from scipy.integrate import solve_ivp
from markov_builder.example_models import construct_wang_chain
from markov_builder import MarkovChain
from markov_builder.rate_expressions import negative_rate_expr, positive_rate_expr

import os
import numpy as np
import myokit as mk
import sympy as sp

from numba import njit

tol = 1e-6


def main():
    mc = construct_wang_chain()

    drugged_states = ["d_O"]

    for s in drugged_states:
        mc.add_state(s)

    rates = [
        ("O", "d_O", "D_on", "D_off"),
    ]

    constant_rate_expr = ('a', ('a',))
    rate_dictionary = {'a_a0': positive_rate_expr + ((0.022348, 0.01176),),
                       'a_a1': positive_rate_expr + ((0.013733, 0.038198),),
                       'b_a0': negative_rate_expr + ((0.047002, 0.0631),),
                       'b_a1': negative_rate_expr + ((0.0000689, 0.04178),),

                       # Using 2mmol KCl values
                       'a_1': positive_rate_expr + ((0.090821, 0.023391),),
                       'b_1': negative_rate_expr + ((0.006497, 0.03268),),

                       'k_f': constant_rate_expr + ((0.023761,),),
                       'k_b': constant_rate_expr + ((0.036778,),),
                       }

    new_rate_dictionary = {
        "D_on": ("D * k_on",) + (tuple(),),
        "D_off": ("D * k_off",) + (tuple(),),
    }

    rate_dictionary = {**new_rate_dictionary, **rate_dictionary}
    print(rate_dictionary)

    for r in rates:
        mc.add_both_transitions(*r)

    shared_variable_dict = {
        "k_on": 1e-2,
        "k_off": 1e-1,
    }

    print(mc.default_values)

    mc.parameterise_rates(rate_dictionary, shared_variables=shared_variable_dict)

    labels = mc.get_states()
    A, B =  mc.eliminate_state_from_transition_matrix(labels[:-1],
                                                      use_parameters=True)

    print(A)
    print(B)

    parameter_labels = sorted([key
                               for key, val in list(mc.default_values.items())
                               if str(key) not in ['E_Kr', 'E_rev', 'V']
                               and val is not None])
    print(parameter_labels)

    param_values = np.array([mc.default_values[k] for k in parameter_labels])

    # Setup protocol
    mk_protocol = mk.load_protocol("simplified-staircase.mmt")
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

    # Now repeat protocol 3 times
    offset = protocol[-1, 1]
    protocol = np.vstack([protocol, protocol + np.array([[offset, offset, 0.0, 0.0]]),
                          protocol + np.array([[offset, offset, 0.0, 0.0]])])

    # Define inputs for sympy function
    D_symbol = "D"

    y_symbols = sp.Matrix([mc.get_state_symbol(s) for s in labels][:-1])
    p_symbols = sp.Matrix([key
                           for key, val in mc.default_values.items()
                           if str(key) not in ['E_Kr', 'E_rev', 'V']
                           and val is not None])

    v_symbol = "V"

    inputs = (y_symbols, p_symbols, v_symbol, D_symbol)
    rhs_expr = A @ y_symbols + B
    rhs_expr = rhs_expr.subs(mc.rate_expressions)

    rhs_func = njit(sp.lambdify(inputs, rhs_expr))

    y0 = np.array([0 for y in y_symbols])
    y0[0] = 1.0
    print("rhsfunc")
    val = rhs_func(y0, param_values, 0.0, 0.0)
    print(val)

    @njit
    def f_deriv(t, y, p=param_values, offset=0.0):
        p = p.copy().flatten()
        y = y.flatten()

        v = protocol_func(t, offset=offset, protocol=protocol)
        D = drug_func(t, offset=offset, protocol=protocol)

        dy = rhs_func(y, p, v, float(D)).flatten()
        return dy

    A_func = sp.lambdify(inputs, A)
    def jac_func(t, y, p=param_values, offset=0.0):
        offset = p
        p = p.copy()
        v = protocol_func(t, offset=offset, protocol=protocol)
        return A_func(y, p, v).flatten()

    sol = solve_ivp(f_deriv, (-1e4, 1e-5), y0, atol=tol, rtol=tol,
                    method='BDF', dense_output=True, #jac=jac_func,
                    args=(param_values,))

    y0 = sol.y[:, -1].flatten()

    # Solve over each step of the protocol
    count = 0
    res = []
    print(protocol)
    for step in protocol:
        tstart, tend, vstart, vend = step
        if tstart == tend:
            continue
        ts = sol.t
        print("times", ts)
        _ts = ts[(ts >= tstart) & (ts <= tend)]


        if tend not in _ts:
            _ts = np.append(_ts, tend)

        y = solve_ivp(f_deriv, (tstart, tend), y0, args=(param_values,), atol=tol,
                      rtol=tol, method='BDF', dense_output=True,
                      # jac=jac_func
                      )

        y0 = y.y[:, -1].flatten()
        count += y.nfev

        ys = y.sol(ts)
        if len(ys) > 1:
            res.append(ys.T[:-1, :])

holding_potential = -80.0
@njit
def protocol_func(t, offset, protocol):
    t = t + offset
    if t < 0 or t >= protocol[-1][1]:
        return holding_potential

    for i in range(len(protocol)):
        if t <= protocol[i][1]:
            if np.abs(protocol[i][3] - protocol[i][2]) > 0.0:
                return protocol[i][2] + (t - protocol[i][0])*(protocol[i][3]-protocol[i][2])/(protocol[i][1] - protocol[i][0])
            else:
                return protocol[i][3]


@njit
def drug_func(t, offset, protocol):
    if t > protocol[:, 1].max() / 3 and t < 2 * protocol[:, 1].max()/3:
        return 1.0
    return 0.0


if __name__ == "__main__":
    main()
