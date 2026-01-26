import matplotlib.pyplot as plt
import numpy as np
import math

def plot_value_grid(V, n_rows, n_cols, title=""):
    grid = np.zeros((n_rows, n_cols), dtype=float)
    for r in range(n_rows):
        for c in range(n_cols):
            grid[r, c] = V[(r, c)]
    
    fig, ax = plt.subplots()
    ax.imshow(grid)

    for r in range(n_rows):
        for c in range(n_cols):
            ax.text(c, r, f"{grid[r,c]:.2f}", ha="center", va="center")

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_title(title)
    plt.show()

def plot_selected_ks(history, ks, n_rows, n_cols):
    max_k = len(history) - 1
    for k in ks:
        if not (0 <= k <= max_k):
            raise ValueError(f"k={k} out of range (0..{max_k})")
        plot_value_grid(history[k], n_rows, n_cols, title=f"Iterative Policy Eval — sweep k = {k}")

def plot_values_by_policy_iteration(log, env):
    for i, V in enumerate(log["values_by_iter"]):
        plot_value_grid(
            V,
            env.n_rows,
            env.n_cols,
            title=f"Policy iteration i = {i}"
        )

def plot_eval_sweeps_for_policy(log, env, policy_iter, ks):
    history = log["eval_histories"][policy_iter]
    plot_selected_ks(
        history,
        ks,
        env.n_rows,
        env.n_cols
    )

ARROWS = {
    (-1, 0): "↑",
    ( 1, 0): "↓",
    ( 0,-1): "←",
    ( 0, 1): "→",
}

def plot_policy(policy_snapshot, env, title=""):
    fig, ax = plt.subplots()
    ax.set_xlim(-0.5, env.n_cols - 0.5)
    ax.set_ylim(env.n_rows - 0.5, -0.5)

    for r in range(env.n_rows):
        for c in range(env.n_cols):
            s = (r, c)
            if s not in policy_snapshot:
                continue

            arrows = ""
            for a, p in policy_snapshot[s].items():
                if p > 1e-12:
                    arrows += ARROWS[a]

            ax.text(c, r, arrows, ha="center", va="center", fontsize=14)

    ax.set_xticks(range(env.n_cols))
    ax.set_yticks(range(env.n_rows))
    ax.set_title(title)
    ax.grid(True)
    plt.show()

def _infer_capacity_from_env(env):
    return int(env.max_capacity)


def plot_jacks_policy(policy_snapshot, env, title=None, show_numbers=True):
    cap = _infer_capacity_from_env(env)
    n = cap + 1

    A = [[0 for _ in range(n)] for __ in range(n)]
    for i in range(n):
        for j in range(n):
            dist = policy_snapshot[(i, j)]
            # choose the most-probable action (should be deterministic after improvement)
            a_star = max(dist.items(), key=lambda kv: kv[1])[0]
            A[i][j] = a_star

    plt.figure()
    plt.imshow(A, origin="lower", aspect="equal")
    plt.colorbar(label="cars moved overnight (a)")
    plt.xlabel("cars at location 2")
    plt.ylabel("cars at location 1")
    if title:
        plt.title(title)

    if show_numbers:
        for i in range(n):
            for j in range(n):
                plt.text(j, i, str(A[i][j]), ha="center", va="center", fontsize=6)

    plt.tight_layout()
    plt.show()


def plot_jacks_value(V, env, title=None, show_numbers=False):
    """
    V: dict { state_tuple: value }
    env: JacksCarRental
    """
    cap = _infer_capacity_from_env(env)
    n = cap + 1

    grid = [[0.0 for _ in range(n)] for __ in range(n)]
    for i in range(n):
        for j in range(n):
            grid[i][j] = float(V[(i, j)])

    plt.figure()
    plt.imshow(grid, origin="lower", aspect="equal")
    plt.colorbar(label="V(s)")
    plt.xlabel("cars at location 2")
    plt.ylabel("cars at location 1")
    if title:
        plt.title(title)

    if show_numbers:
        for i in range(n):
            for j in range(n):
                plt.text(j, i, f"{grid[i][j]:.0f}", ha="center", va="center", fontsize=6)

    plt.tight_layout()
    plt.show()

