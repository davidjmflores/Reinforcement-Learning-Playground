import matplotlib.pyplot as plt
import numpy as np

def plot_value_grid(V, n_rows, n_cols, title=""):
    grid = np.zeros((n_rows, n_cols), dtype=float)
    for r in range(n_rows):
        for c in range(n_cols):
            grid[r, c] = V[(r, c)]
    
    fig, ax = plt.subplots()
    ax.imshow(grid)

    # add values to boxes
    for r in range(n_rows):
        for c in range(n_cols):
            ax.text(c, r, f"{grid[r,c]:.2f}", ha="center", va="center")

    ax.set_xticks(range(n_cols))
    ax.set_yticks(range(n_rows))
    ax.set_title(title)
    plt.show()

def plot_selected_ks(history, ks, n_rows, n_cols):
    for k in ks:
        plot_value_grid(history[k], n_rows, n_cols, title=f"Iterative Policy Eval — sweep k = {k}")
