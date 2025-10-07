import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import markov_builder
import matplotlib
import cma
from scipy.integrate import solve_ivp
from markov_builder.example_models import construct_wang_chain, construct_four_state_chain
from markov_builder import MarkovChain
from markov_builder.rate_expressions import negative_rate_expr, positive_rate_expr

from numbalsoda import lsoda, lsoda_sig

import os
import numpy as np
import myokit as mk
import sympy as sp
import seaborn as sns

from numba import njit

tol = 1e-12


def main():

    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--output_dir", default='output')
    arg_parser.add_argument("--figsize", default=[4.8, 5.125], type=float,
                            nargs=2)

    global args
    args = arg_parser.parse_args()

    output_dir = os.path.join(args.output_dir,
                              "simple_drug_model")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    mc = construct_wang_chain()

    drugged_states = ["d_O"]

    for s in drugged_states:
        mc.add_state(s)

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
        "D_off": ("k_off",) + (tuple(),),
    }

    rate_dictionary = {**new_rate_dictionary, **rate_dictionary}
    print(rate_dictionary)

    rates = [
        ("O", "d_O", "D_on", "D_off"),
    ]

    for r in rates:
        mc.add_both_transitions(*r)

    shared_variable_dict = {
        "k_on":  1e-1,
        "k_off": 1e-2,
    }

    print(mc.default_values)

    mc.parameterise_rates(rate_dictionary, shared_variables=shared_variable_dict)

    states = sorted(mc.get_states())

    # Reverse order of closed states so that they're the right way round in the legend
    states = list(reversed(["d_O", "O", "I", "C1", "C2", "C3"]))

    state_labels, Q =  mc.get_transition_matrix(use_parameters=True, label_order=states)

    parameter_labels = sorted([key
                               for key, val in list(mc.default_values.items())
                               if str(key) not in ['E_Kr', 'E_rev', 'V', 'D', 'g_Kr']
                               and val is not None])
    print(state_labels, Q)
    print(parameter_labels)

    param_values = np.array([mc.default_values[k] for k in parameter_labels])

    # Setup protocol
    mk_protocol = mk.load_protocol("simplified-staircase.mmt")
    protocol = []
    t_cur = 0

    protocol = np.array([[0, 5000.0, 0.0, 0.0],
                         [5000.0, 10000.0, 0.0, 0.0],
                         [10000.0, 19999.0, 0.0, 0.0],
                         [19999.0, 30000.0, 0.0, 0.0]
                         ])

    protocol = np.vstack(protocol).astype(np.float64)

    # Define inputs for sympy function
    D_symbol = "D"

    y_symbols = sp.Matrix([mc.get_state_symbol(s) for s in state_labels])
    p_symbols = parameter_labels

    v_symbol = "V"

    inputs = (y_symbols, p_symbols, v_symbol, D_symbol)
    # rhs_expr = A @ y_symbols + B

    rhs_expr = Q.T @ y_symbols
    rhs_expr = rhs_expr.subs(mc.rate_expressions)

    print(state_labels)
    print(rhs_expr)

    rhs_func = njit(sp.lambdify(inputs, rhs_expr))

    y0 = np.array([0 for y in y_symbols])
    y0[0] = 1.0
    val = rhs_func(y0, param_values, 0.0, 0.0)
    print("rhsfunc", val)

    @njit
    def f_deriv(t, y, p=param_values, offset=0.0):
        p = p.copy().flatten()
        y = y.flatten()

        v = protocol_func(t, offset=offset, protocol=protocol)
        D = drug_func(t, offset=offset, protocol=protocol)

        dy = rhs_func(y, p, v, float(D)).flatten()
        return dy

    sol = solve_ivp(f_deriv, (-1e4, 0), y0, atol=tol, rtol=tol,
                    dense_output=True, #jac=jac_func,
                    args=(param_values,))

    y0 = sol.y[:, -1].flatten()
    print("Sol at t=0: ", y0)

    # Solve over each step of the protocol
    count = 0
    res = []
    ts = np.linspace(protocol[0, 0], protocol[-1, 1],
                     int(protocol[-1, 0]) * 10)

    ys = []
    for step in protocol:
        tstart, tend, vstart, vend = step
        _ts = ts[(ts >= tstart) & (ts <= tend)]

        if tend not in _ts:
            _ts = np.append(_ts, tend)

        y0 = y0.flatten()

        y = solve_ivp(f_deriv, (tstart, tend), y0, args=(param_values,), atol=tol,
                      rtol=tol, dense_output=False, t_eval=_ts)

        print(y)

        _ys = y.y.T
        y0 = _ys[-1, :].flatten()

        print(tstart, tend, _ts)
        print(_ys)
        ys.append(_ys[:-1, :])

    # Add last observation
    ys.append(y0.flatten()[None, :])

    ys = np.vstack(ys)

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)

    axs = fig.subplots(2, 1, sharex=True, height_ratios=[.5, 1])
    Ds = np.array([drug_func(t, 0.0, protocol) for t in ts])

    axs[0].plot(ts, Ds, color="black")
    print(protocol)
    Vs = np.array([protocol_func(t, 0.0, protocol) for t in ts])
    # axs[1].plot(ts, Vs, color="black")
    # axs[1].plot(ts, ys, label=state_labels)

    occupations_ax = axs[1]
    culm_states = np.full(ys.shape[0], 0.0)
    colours = sns.husl_palette(len(state_labels))

    state_label_dict = {s: r"$" f"{s[0]}_{s[1:]}" r"$" if len(s) > 1
                        else r"$" f"{s[0]}" r"$"
                        for s in state_labels}

    state_label_dict["d_O"] = r"$O_D$"

    for i in range(ys.shape[1]):
        colour = colours[i]
        label = state_label_dict[state_labels[i]]

        occupations_ax.plot(ts, culm_states + ys[:, i].flatten(),
                            color='grey', lw=1.0)

        occupations_ax.fill_between(ts, culm_states,
                                    culm_states + ys[:, i].flatten(),
                                    color=colour,
                                    label=label)

        culm_states += ys[:, i].flatten()

    handles, labels = axs[1].get_legend_handles_labels()
    axs[1].legend(handles[::-1], labels[::-1], frameon=True)


    for ax in axs:
        for side in [["top", "right"]]:
            ax.spines[side].set_visible(False)

    for ax in axs[:-1]:
        ax.set_xticklabels([])


    axs[1].set_xlim(0, ts.max())
    axs[1].set_ylim(0, 1.0)

    xticks = axs[1].get_xticks()
    axs[1].set_xticks(xticks)

    axs[1].set_xticklabels([float(x) * 1e-3 for x in xticks])
    axs[1].set_xlabel(r"$t$ (s)")

    axs[0].set_ylabel(r"$D$")
    axs[1].set_ylabel(r"State occupancy")

    for ax, cap in zip(axs, "abcdef"):
        ax.set_title(cap, loc="left", weight="bold")

    fig.savefig(os.path.join(output_dir, "drug_model_output.pdf"))


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
    return holding_potential


@njit
def drug_func(t, offset, protocol):
    t += offset
    if t > protocol[1, 0] and t < protocol[2, 1]:
        return 1.0
    return 0.0


if __name__ == "__main__":
    main()
