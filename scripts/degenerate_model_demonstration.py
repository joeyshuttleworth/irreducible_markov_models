from auxin_signalling_pathway.reduced_auxin_signalling_pathway import \
    reduced_auxin_signalling_pathway as reduced_asp_class
from auxin_signalling_pathway import auxin_signalling_pathway as asp_class
import matplotlib.gridspec as gridspec
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import sympy as sp
import seaborn as sns
import scipy
import os
import argparse

from numba import njit


font = {"size": 9}

mpl.rc('font', **font)


def main():

    kwargs = {'nARFs': 1, 'nIAAs': 1}

    arg_parser = argparse.ArgumentParser("description")
    arg_parser.add_argument("-o", "--output_dir", type=str, default="output")
    arg_parser.add_argument("--figsize",  type=float, nargs=2, default=[6, 5],)

    # Plot IAA RNA
    plot_var_name = "R__iaa_1"

    global args
    args = arg_parser.parse_args()

    fig = plt.figure(constrained_layout=True, figsize=args.figsize)
    gs = gridspec.GridSpec(4, 3, figure=fig,
                           height_ratios=[0.4, 1, 1, 1.5],
                           width_ratios=[1, 1, 0.05])
    legend_ax = fig.add_subplot(gs[0, :])
    solution_ax = fig.add_subplot(gs[1, :-1])
    convergence_ax = fig.add_subplot(gs[2, :-1])
    scatter_steps_ax = fig.add_subplot(gs[3, 0])
    scatter_rmse_ax = fig.add_subplot(gs[3, 1])

    cax = fig.add_subplot(gs[1:3, -1])

    legend_ax.axis("off")

    axs = [solution_ax, convergence_ax, scatter_steps_ax, scatter_rmse_ax]
    for ax in axs:
        for side in ["top", "right"]:
            ax.spines[side].set_visible(False)

    solution_ax.set_title("a", loc="left", weight="bold")
    convergence_ax.set_title("b", loc="left", weight="bold")
    scatter_steps_ax.set_title("c", loc="left", weight="bold")
    scatter_rmse_ax.set_title("d", loc="left", weight="bold")

    global output_dir
    output_dir = os.path.join(args.output_dir, "degenerate_model_demo")

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    asp_model_r = reduced_asp_class(**kwargs, include_arf_transcription=True)
    asp_model_full = asp_class(**kwargs, include_arf_transcription=True)
    asp_model_full.remove_state('G__ARF_1')

    plot_var_index_red = asp_model_r.get_state_variables().index(plot_var_name)
    plot_var_index_full = asp_model_full.get_state_variables().index(plot_var_name)

    # Find all parameters related to the promoter-protein dynamics
    empty_set_node = asp_model_full.empty_set_node
    aux_symbol = asp_model_full.aux_symbol

    pp_reactions = []
    promoter_symbols = [s for s in asp_model_full.get_state_variables() if s[0]=='G']
    for r in asp_model_full._reactions_set:
        if np.all(np.isin(r.reactants + r.products, promoter_symbols)):
            pp_reactions.append(r)

    pp_parameters = [s for r in pp_reactions for s in r.fwd_rates + r.bwd_rates]
    print(asp_model_full.get_default_parameters())
    print(pp_parameters)

    promoter_timeconstant = r"tau_G"
    asp_model_full.add_transition_rate(promoter_timeconstant, 1.0, "Promoter timeconstant", strict=False)
    asp_model_full.add_transition_rate("k__RNA__IAA", 1.0, "Rate of IAA production from RNA")
    asp_model_full.add_transition_rate("k__RNA__ARF", 1.0, "Rate of ARF production from RNA")

    full_parameter_subs_dict = asp_model_r.parameter_subs_dict.copy()
    asp_model_full.remove_transition_rate('k__R__iaa_1__iaa_1')

    for p in pp_parameters:
        if p in full_parameter_subs_dict:
            print(f"{full_parameter_subs_dict[p]}")
        else:
            full_parameter_subs_dict[p] = sp.sympify(p) / sp.sympify(promoter_timeconstant)

    # Generate initial conditions
    reduced_ic_dict = asp_model_r.get_default_initial_conditions()
    full_ic_dict = asp_model_full.get_default_initial_conditions()

    jac_func, inhomog_func = asp_model_r.get_promoter_ode_funcs(njitted=True)

    # Randomise parameters slightly
    reduced_param_dict = asp_model_r.get_default_parameters()
    for k in reduced_param_dict:
        reduced_param_dict[k] *= 10**np.random.uniform(0, .1)


    new_params = {}
    new_params["k__D__ARF_1__ARF_1"] = 5e-2
    new_params["d__D__ARF_1__ARF_1"] = 1e-2

    new_params["k__D__ARF_1__iaa_1"] = 5e-2
    new_params["d__D__ARF_1__iaa_1"] = 1e-2

    new_params["k__D__iaa_1__iaa_1"] = 5e-2
    new_params["d__D__iaa_1__iaa_1"] = 1e-2

    new_params["d__R"] = 1e-1
    new_params["k__RNA__IAA"] = 1.0
    new_params["k__RNA__ARF"] = 1.0

    new_params["d__iaa_1__x"] = 0.0

    new_params["k__G__ARF_1__ARF_1__R__iaa_1"] = 1.0

    new_params["d__arf__all"] = 2.0
    new_params["d__iaa__all"] = 1.0

    new_params["k__G__ARF_1__iaa_1__R__ARF_1"] = 0.0
    new_params["k__G__ARF_1__ARF_1__R__ARF_1"] = 1.0
    new_params["k__G__R__ARF_1"] = 0.0001

    new_params["k__G__ARF_1__ARF_1__R__iaa_1"] = 1.0
    new_params["k__G__ARF_1__iaa_1__R__iaa_1"] = 0.0
    new_params["k__G__R__iaa_1"] = 0.0001

    for k,v in new_params.items():
        if k in reduced_param_dict:
            reduced_param_dict[k] = v
        else:
            raise Exception(f"param {k} not present in model")

    reduced_ic_dict = asp_model_r.get_default_initial_conditions()
    full_param_dict = reduced_param_dict.copy()
    odes, free_variables, transition_rates, constraints = \
        asp_model_r.generate_ode_system(reduced=True)

    params = np.array(list(full_param_dict.values()))

    jac = jac_func(np.array(list(reduced_ic_dict.values())).astype(np.float64),
                params,
                np.array([1.0]))

    inhomog = inhomog_func(np.array(list(reduced_ic_dict.values())),
                            params, [1.0])

    initial_promoter_vars = np.linalg.solve(jac, -inhomog).flatten()
    initial_promoter_vars = np.concatenate([[1.0 - initial_promoter_vars.sum()], initial_promoter_vars])

    # We can set the initial conditions to be identical, but this produces less interesting plots:
    for i, p in enumerate(promoter_symbols):
        # break
        full_ic_dict[p] = initial_promoter_vars[i]

    reduced_solver = asp_model_r.make_forward_solver()
    ts = np.linspace(0, 500.0, 1000)

    sol1, succ = reduced_solver(ts, reduced_param_dict, reduced_ic_dict, 1)
    r_odes, free_variables, transition_rates, constraints = asp_model_r.generate_ode_system()
    f_odes, free_variables, transition_rates, constraints = asp_model_full.generate_ode_system(parameter_subs_dict=full_parameter_subs_dict)

    relabel_dict = {
        "ARF_1": r"$A$",
        "iaa_1": r"$I$",
        "D__ARF_1__ARF_1": r"$D_{A, A}$",
        "D__ARF_1__iaa_1": r"$D_{A,I}$",
        "D__iaa_1__iaa_1": r"$D_{I,I}$",
        "R__iaa_1": r"$R_I$",
        "R__ARF_1": r"$R_A$",
    }
    labels = [relabel_dict[k] if k in relabel_dict else k
              for k in reduced_ic_dict.keys()]

    states_included_indices = [i for i in range(len(labels))
                               if labels[i][:3] != r"$R_"]

    labels = [labels[i] for i in states_included_indices]

    solution_ax.plot(ts, sol1[:, states_included_indices],
                     label=labels)

    # solution_ax.legend(ncol=3, fontsize=8)
    h, l = solution_ax.get_legend_handles_labels()

    leg = legend_ax.legend(h, l, loc="center", ncol=3, fontsize=9)

    leg.set_frame_on(False)

    full_param_dict = asp_model_full.get_default_parameters(parameter_subs_dict=asp_model_r.parameter_subs_dict)

    for k, v in reduced_param_dict.items():
        full_param_dict[k] = v

    full_param_dict[promoter_timeconstant] = 1e-10

    full_solver = asp_model_full.make_forward_solver(parameter_subs_dict=full_parameter_subs_dict)
    sol2, succ = full_solver(ts, full_param_dict, full_ic_dict, 1, strict=False)

    print(full_ic_dict.keys())
    fig2 = plt.figure()
    ax = fig2.subplots()
    ax.plot(ts, sol2 / sol2[-1, :], label=full_ic_dict.keys())
    ax.legend()

    odes, free_variables, transition_rates, \
                _ = asp_model_r.generate_ode_system(
                                            reduced=True)

    cmap = sns.color_palette('rocket_r', as_cmap=True)

    fig2.savefig(os.path.join(output_dir, "example_solution_full_model.pdf"))
    plt.close(fig2)

    rmses = []
    tau_vals = 10**np.linspace(0, -2, 9)

    parameter_subs_dict = asp_model_r.parameter_subs_dict
    full_param_dict = asp_model_full.get_default_parameters(parameter_subs_dict=parameter_subs_dict)

    for k, v in reduced_param_dict.items():
        full_param_dict[k] = v

    parameter_subs_dict = asp_model_r.parameter_subs_dict
    for i, tau in enumerate(tau_vals):
        # full_param_dict = asp_model_full.get_default_parameters(parameter_subs_dict=parameter_subs_dict)
        full_param_dict[promoter_timeconstant] = tau
        sol2, succ = full_solver(ts, full_param_dict, full_ic_dict, 1)
        color = cmap(i / len(tau_vals))
        convergence_ax.plot(ts, sol2[:, plot_var_index_full], color=color)

        rmse = np.sqrt(np.mean((sol2[1:, plot_var_index_full] - sol1[1:, plot_var_index_red])**2))
        rmses.append(rmse)

    # divider = make_axes_locatable(solution_ax)
    # _cax = divider.append_axes('right', size='10%', pad=0.05)
    # _cax.set_visible(False)
    # divider = make_axes_locatable(convergence_ax)
    # cax = divider.append_axes('right', size='10%', pad=0.05)
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=mpl.colors.LogNorm(tau_vals.min(), tau_vals.max()),
                                       cmap='rocket'), cax=cax,
                        orientation='vertical', ax=plt.gca(), shrink=1.0, pad=0.0)
    cbar.ax.tick_params(labelsize=9)  # Set tick label font size to 14

    cbar.ax.set_ylabel(r"$\tau_G$")
    cbar.ax.set_aspect("auto")
    cbar.ax.xaxis.set_label_position("bottom")

    # cax.set_axis_off()
    for side in cax.spines:
        cax.spines[side].set_visible(False)

    convergence_ax.plot(ts, sol1[:, plot_var_index_red], color='grey', ls='--')

    convergence_ax.set_ylabel(r"$R_I$")

    convergence_ax.set_xlabel(r"$t$ (s)")

    cmap = sns.color_palette('rocket_r', as_cmap=True)
    steps_taken_list = []
    x0 = np.array(list(full_ic_dict.values()))
    p = np.array(list(full_param_dict.values()))

    _ts = [ts[0], ts[-1]]
    parameter_subs_dict = full_parameter_subs_dict
    print(parameter_subs_dict)
    _rhs_func = asp_model_full.generate_rhs_func(parameter_subs_dict=parameter_subs_dict, njitted=True)

    def wrapped_rhs_func(t, y, p):
        return _rhs_func(y, p, (1.0,)).flatten()

    for i, tau in enumerate(tau_vals):
        print(tau)
        full_param_dict[promoter_timeconstant] = tau
        p = np.array(list([full_param_dict[k] for k in sorted(full_param_dict.keys())]))
        sol = scipy.integrate.solve_ivp(wrapped_rhs_func, _ts, x0, args=(p,))
        steps_taken = sol.nfev
        steps_taken_list.append(steps_taken)

    # Now the reduced model
    x0 = np.array(list(reduced_ic_dict.values()))
    p = np.array(list(reduced_param_dict.values()))

    _rhs_func = asp_model_r.generate_rhs_func(parameter_subs_dict)
    pp_jac_func, pp_inhomog_func = asp_model_r.get_promoter_ode_funcs()

    print("Solving degenerate model")

    def wrapped_rhs_func(t, y, p):
        pp_inhomog = pp_inhomog_func(y, p, [1.0])

        pp_jac = pp_jac_func(y, p, [1.0])
        pp_inhomog = pp_inhomog_func(y, p, [1.0])
        state_ss = -np.linalg.solve(pp_jac, pp_inhomog).flatten()
        G = 1.0 - np.sum(state_ss)

        promoter_vars = np.concatenate([np.array([G]), state_ss])
        return _rhs_func(y, p, [1.0], promoter_vars).flatten()

    print("Counting solver steps for degenerate system")
    sol = scipy.integrate.solve_ivp(wrapped_rhs_func, y0=x0, t_span=_ts, args=(p,))

    steps_taken = sol.nfev

    ms = 10
    scatter_steps_ax.scatter(tau_vals, steps_taken_list, marker='x', s=ms)
    scatter_steps_ax.axhline(steps_taken, ls='--', color='grey')
    scatter_steps_ax.set_xscale('log')
    scatter_steps_ax.set_yscale('log')
    scatter_steps_ax.set_ylabel("RHS evaluations")
    scatter_steps_ax.set_xlabel(r"$\tau_G$")

    scatter_rmse_ax.scatter(tau_vals, rmses, marker='x', s=ms)
    scatter_rmse_ax.set_xscale('log')
    scatter_rmse_ax.set_yscale('log')
    scatter_rmse_ax.set_xlabel(r"$\tau_G$")
    scatter_rmse_ax.set_ylabel(r"RMSE")

    fig.savefig(os.path.join(output_dir, "degenerate_model_fig.pdf"))

if __name__ == "__main__":
    main()
