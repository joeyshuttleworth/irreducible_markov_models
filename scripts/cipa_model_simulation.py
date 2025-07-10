import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import markov_builder
import matplotlib
import cma
from scipy.integrate import solve_ivp
# from markov_builder.example_models import construct_kemp_model
# from markov_builder.example_models import construct_HH_model
from markov_builder import MarkovChain
from markov_builder.rate_expressions import negative_rate_expr, positive_rate_expr

import os
import numpy as np


def main():
    mc = MarkovChain(name="CiPA Model")

    states = ["O", "I", "IC", "C", "IC2", "C2"]
    drugged_states = ["d_IO", "d_O", "d_C"]

    for s in states:
        mc.add_state(s)

    for s in drugged_states:
        mc.add_state(s)

    rates = [
        ("O", "C", "b_2", "a_2"),
        ("I", "IC", "b_2", "a_2"),
        ("C", "C2", "b_1", "a_1"),
        ("IC", "IC2", "b_1", "a_1"),
        ("O", "I", "b_h", "a_h"),
        ("C", "IC", "b_h", "a_h" ),
        ("C2", "IC2", "b_h", "a_h")
    ]

    # Baseline model
    rate_dictionary = {
        # Activation rates
        'a_1': positive_rate_expr + ((8.53e-03, 8.32e-02),),
        'a_2': positive_rate_expr + ((1.49e-01, 2.43e-02),),

        # Deactivation rates
        'b_1': negative_rate_expr + ((1.26e-02, 1.04e-04),),
        'b_2': negative_rate_expr + ((5.58e-04, 4.07e-02),),

        # Recovery rate
        'a_h': negative_rate_expr + ((7.67e-02, 2.25e-02),),

        # Inactivation rate
        'b_h': positive_rate_expr + ((2.70e-01, 1.58e-02),),
    }

    for r in rates:
        mc.add_both_transitions(*r)

    rates = [
        ('O', 'd_O', 'drug_on', 'drug_off'),
        ('I', 'd_IO', "drug_on", "0"),
        ('d_O', 'd_C', 'trap', 'untrap'),
        ('d_IO', 'd_C', 'trap', 'untrap'),
    ]

    for r in rates:
        mc.add_both_transitions(*r)

    new_rate_dict = {"drug_on": ["K_u * K_max * D**n / (D**n + halfmax)", tuple()],
                      "drug_off": [r"K_u", tuple()],
                      "trap": ["Kt", tuple()],
                      "untrap": ["Kt / (1 + exp(-(V - Vhalf) / 6.789))", tuple()]
                      }
    new_rate_dict = {**rate_dictionary, **new_rate_dict}

    shared_variable_dict = {"K_u": 1e-3,
                            "Kt": 3.5e-5,
                            "K_max": 1e-3,
                            "n": 1,
                            "halfmax": 1,
                            "Vhalf": 0
                            }
    mc.parameterise_rates(new_rate_dict, shared_variables=shared_variable_dict)

    labels = mc.get_states()
    A, B =  mc.eliminate_state_from_transition_matrix(labels[:-1],
                                                      use_parameters=True)

    print(A)
    print(B)

    parameter_labels = sorted([key
                               for key, val in list(mc.default_values.items())
                               if str(key) not in ['E_Kr', 'E_rev', 'V']
                               and val is not None])

    param_values = [mc.default_values[k] for k in parameter_labels]

    print(parameter_labels)
    print(param_values)
    # Setup protocol
    mk_protocol = mk.load_protocol("simplified-staircase.mmt")
    protocol = []
    t_cur = 0

    for event in mk_protocol.events():
        duration = event.duration()
        start_t = t_cur
        end_t = start_t + duration
        t_cur = end_t
        level = event.leve
l()
        protocol.append((start_t, end_t, level, level))

    protocol = np.vstack(protocol).astype(np.float64)

    # Now repeat protocol 3 times
    offset = protocol[-1, 1]
    protocol = np.vstack([protocol, protocol + np.array([[offset, offset, 0.0, 0.0]]),
                          protocol + np.array([[offset, offset, 0.0, 0.0]])])

    # Define inputs for sympy function
    inputs = (y_symbols, p_symbols, v_symbol, D_symbol)

    rhs_expr = A @ y_symbols + B
    rhs_func = sp.lambdify(inputs, rhs_func)

    @njit
    def deriv_func(t, y, p):
        offset = p[-1]
        p = p[:-1].copy()
        v = protocol_func(t, offset=offset, protocol=protocol)
        D = drug_func(t, offset=offset, protocol=protocol)

        return rhs_func(y, p, v, D)

    A_func = sp.lambdify(inputs, A)
    def jac_func(t, y, p=mm.default_parameters):
        offset = p[-1]
        p = p[:-1].copy()
        v = voltage_func(t, offset=offset)
        return A_func(y, p, v)

    param_values = np.append(param_values, 0.0)

    sol = solve_ivp(f_deriv, (-1e4, 1e-5), y0, args=(p,), atol=tol,
                    rtol=tol, method='BDF', dense_output=True, jac=jac_func)
    y0 = sol.y[:, -1].flatten()

    # Solve over each step of the protocol
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
        return 1
    return 0


if __name__ == "__main__":
    main()
