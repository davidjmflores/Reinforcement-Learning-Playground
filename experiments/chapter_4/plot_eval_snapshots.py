import matplotlib.pyplot as plt
import numpy as np

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
