import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import argparse
import markov_builder
import matplotlib
import cma
from scipy.integrate import solve_ivp

import os
import numpy as np

colours = matplotlib.cm.Set2(range(4))

def create_axs(fig):
    n_models = 3

    gs = gridspec.GridSpec(3, 2, width_ratios=[2, 1],
                           figure=fig)

    # Left column with shared x-axis
    ax_left_top = fig.add_subplot(gs[0, 0])
    ax_left_middle = fig.add_subplot(gs[1, 0], sharex=ax_left_top)
    ax_left_bottom = fig.add_subplot(gs[2, 0], sharex=ax_left_top)

    # Right column (no sharing needed)
    ax_right_top = fig.add_subplot(gs[0, 1])
    ax_right_middle = fig.add_subplot(gs[1, 1])
    ax_right_bottom = fig.add_subplot(gs[2, 1])

    for ax in [ax_right_top, ax_right_middle, ax_right_bottom]:
        ax.set_axis_off()

    for ax in [ax_left_top, ax_left_middle, ax_left_bottom]:
        for side in ['top', 'right']:
            ax.spines[side].set_visible(False)

        # ax.set_ylim([0.0, 1.0])

    # Hide x-axis labels except the bottom left
    plt.setp(ax_left_top.get_xticklabels(), visible=False)
    plt.setp(ax_left_middle.get_xticklabels(), visible=False)

    return np.array([[ax_left_top, ax_right_top],
                     [ax_left_middle, ax_right_middle],
                     [ax_left_bottom, ax_right_bottom]])

def generate_models():

    k1, k2, k3, k4 = [0.5, 1.5, 1.0, 3.0]
    Q1 = np.array([[-k1, k1, 0, 0],
                   [k2, -k2, 0, 0],
                   [0, 0, -k3, k3],
                   [0, 0, k4, -k4]])

    Q2 = np.array([[-11, 1, 10], [2, -5, 3], [5, 4, -9]]) / 10.0

    k1, k2, k3, k4, k5, k6 = np.array([1, 0.1, 1, 0.1, 2, 0.1])

    Q3 = np.array([[-k1-k6, k1, k6], [k2, -k2-k3, k3], [k5, k4, -k4-k5]]).astype(np.float64)

    return Q1, Q2, Q3

def main():


    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument("--figsize", nargs=2, type=float, default=[4.3, 5.0])
    args = arg_parser.parse_args()

    fig = plt.figure(figsize=args.figsize, constrained_layout=True)
    axs = create_axs(fig)

    t_eval = np.linspace(0, 2.5, 500)

    def deriv_func(t, x, Q):
        return Q.T @ x

    def optim_func(xvec):
        k1, k2, k3, k4, k5, k6 = xvec

        penalty = 0
        if np.any(xvec <= 0):
            penalty += np.sum(xvec[xvec<=0]**2) * 1e5
        k1, k2, k3, k4, k5, k6 = xvec

        xvec = np.max(xvec, 0)

        Q3 = np.array([[-k1-k6, k1, k6],
                       [k2, -k2-k3, k3],
                       [k5, k4, -k4-k5]]).astype(np.float64)
        lamb, vs = np.linalg.eig(Q3)
        min_index = np.argmin(lamb)
        vs = vs[[i for i in range(vs.shape[0]) if i != min_index], :]

        return penalty + 1 / (np.linalg.norm(np.real(vs)) / np.linalg.norm(np.imag(vs)))

    es = cma.CMAEvolutionStrategy([1, 1, 1, 1, 1, 1e-5], 1.0)
    res = es.optimize(optim_func).result
    xopt = res.xbest
    xopt = xopt / np.linalg.norm(xopt)
    score = res.fbest

    k1, k2, k3, k4, k5, k6 = xopt
    Q3 = np.array([[-k1-k6, k1, k6],
                   [k2, -k2-k3, k3],
                   [k5, k4, -k4-k5]]).astype(np.float64)
    lamb, vs = np.linalg.eig(Q3)
    min_index = np.argmin(lamb)
    vs = vs[[i for i in range(vs.shape[0]) if i != min_index], :]

    print(xopt, score, lamb)

    Q1, Q2, Q3 = generate_models()

    state_labels = list("ABCDEFG")
    for i, Q in enumerate([Q1, Q2, Q3]):
        N = Q.shape[0]
        x0 = np.full(N, 1.0/N)
        # if i == 2:
        #     x0 = np.array([1.0, 0, 0])

        res = solve_ivp(deriv_func, [0, t_eval[-1]], x0,
                        t_eval=t_eval,
                        args=(Q,), atol=1e-8,
                        rtol=1e-8)

        sol = res.y.T

        sol = sol / sol.sum(axis=1)[:, None]
        for j in range(N):
            axs[i, 0].plot(t_eval, sol[:, j],
                           label=state_labels[j], color=colours[j])

    output_dir = "output"

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    fig.savefig(os.path.join(output_dir, "markov_model_trajectories"))


if __name__ == "__main__":
    main()
