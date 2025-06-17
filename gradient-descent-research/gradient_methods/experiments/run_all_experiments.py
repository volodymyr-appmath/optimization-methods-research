from typing import Callable

import pandas as pd
import sympy as sym

from gradient_methods.experiments.constants import method_name_mapping, en_to_ua_mapping
from gradient_methods.experiments.data_objects import ExperimentConfig, MinimizationResults
from gradient_methods.experiments.functional import run_experiment
from gradient_methods.experiments.test_functions import generate_test_functions


import matplotlib.pyplot as plt
import numpy as np
import os
import pickle

from numpy.linalg import norm
from matplotlib.ticker import MaxNLocator, FuncFormatter


import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from numpy.linalg import norm

import matplotlib as mpl

mpl.rcParams['axes.labelsize'] = 14
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12


def plot_minimization_paths_2d(
        func_numeric: Callable,
        result_no_restart: MinimizationResults,
        result_restart: MinimizationResults,
        x_bounds: np.ndarray,
        y_bounds: np.ndarray,
        save_path=None,
        plot_restart=False,
        func_name=""
):
    """
    Plots the optimization paths of two methods (with and without restart) on a 2D contour plot of the objective function.

    Parameters:
        func_numeric: Callable 2D function (e.g. lambdified sympy expression)
        result_no_restart: MinimizationResults for method without restart
        result_restart: MinimizationResults for method with restart
        save_path: Path to save the plot (if None, will just show)
    """

    history_no_restart = np.array(result_no_restart.history)
    history_restart = np.array(result_restart.history)

    # Create meshgrid around the points
    x_min, x_max = x_bounds
    y_min, y_max = y_bounds

    X, Y = np.meshgrid(np.linspace(x_min, x_max, 300),
                       np.linspace(y_min, y_max, 300))
    Z = np.vectorize(lambda x, y: func_numeric(x, y))(X, Y)

    fig, ax = plt.subplots(figsize=(6, 6))
    CS = ax.contour(X, Y, Z, levels=30, cmap=cm.coolwarm)
    ax.clabel(CS, inline=1, fontsize=8)

    # Plot paths
    if plot_restart:
        ax.plot(history_restart[:, 0], history_restart[:, 1], 'o-', color='orange', label='Траєкторія мінімізації')
    else:
        ax.plot(history_no_restart[:, 0], history_no_restart[:, 1], 'o-', color='blue', label='Траєкторія мінімізації')

    # Mark start
    x0 = result_no_restart.x_start
    ax.plot(x0[0], x0[1], 'go', markersize=8, zorder=8, label='$x^0$ -- стартова точка')
    ax.text(x0[0], x0[1], r'$x^0$', fontsize=12, verticalalignment='top', horizontalalignment='left')

    # Mark predicted minimum
    x_min_pred = result_restart.x_min_pred if plot_restart else result_no_restart.x_min_pred
    ax.plot(x_min_pred[0], x_min_pred[1], 'co', markersize=8, zorder=8, label=r'$\hat{x}$ -- наближення методу')
    ax.text(x_min_pred[0], x_min_pred[1], r'$\hat{x}$', fontsize=12, verticalalignment='top', horizontalalignment='left')

    # Mark minimum
    x_min = result_no_restart.x_min_true
    ax.plot(x_min[0], x_min[1], 'ro', markersize=8, zorder=10, label='$x^*$ -- істинний мінімум')
    ax.text(x_min[0], x_min[1], r'$x^*$', fontsize=12, verticalalignment='bottom', horizontalalignment='right')


    with_restart_text = "(з відновленням)" if plot_restart else "(без відновлення)"

    ax.set_title(f"{result_no_restart.method_name} {with_restart_text} на функції {func_name}", fontsize=14)
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.legend(loc='upper left', framealpha=0.5).set_zorder(11)
    ax.grid(True)
    ax.set_aspect('equal')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=100)
        print(f"📉 Saved 2D path plot: {save_path}")
        plt.close()
    else:
        plt.show()



def run_method_pair_on_function(func, method_name, precision, save_dir="plots2", result_dir="results"):
    """
    Run method with and without restart on a single function.
    Plot and save the result comparison.
    """

    results = []
    for use_restart in [False, True]:
        label = f"{method_name} {'(restart)' if use_restart else '(no restart)'}"

        config = ExperimentConfig(
            function=func["function"],
            x_start=func["x_start"],
            x_min_true=func["x_min"],
            f_min_true=func["f_min"],
            precision=precision,
            line_search_precision=1e-9,
            use_restart=use_restart,
            method_name=method_name,
        )

        result = run_experiment(config)
        results.append(result)

        os.makedirs(result_dir, exist_ok=True)
        with open(os.path.join(result_dir, f"{label.replace(' ', '_')}_{func['name'].replace(' ', '_')}.pkl"), "wb") as f:
            pickle.dump(result, f)

    # Plotting
    os.makedirs(save_dir, exist_ok=True)
    plt.figure(figsize=(13, 6))

    # Header for legend
    header = (
        # f"$x^0 =$ {np.round(results[0].x_start, 6).tolist()}\n"
        # f"$x^* =$ {np.round(results[0].x_min_true, 6).tolist()}\n"
        f"$f^* =$ {round(results[0].f_min_true, int(abs(np.log10(precision))))}\n"
    )

    for res in results:
        x_error = norm(res.x_min_true - res.x_min_pred)
        f_pred = round(res.f_min_pred, int(abs(np.log10(precision))))
        x_error = round(x_error, int(abs(np.log10(precision))))

        label = (
            f"{res.method_name} {'(з відн.)' if res.use_restart else '(без відн.)'}\n"
            f"$||x^* - \\hat{{x}}||$ = {x_error}\n"
            f"$\\hat{{f}}$ = {f_pred}"
        )

        z_order = 8 if res.use_restart else 10
        plt.plot(res.errors, label=label, linestyle='--', marker='o', markersize=7, zorder=z_order)

    # Format axes
    ax = plt.gca()
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"$10^{{{int(y)}}}$"))

    ax.set_xlabel("Ітерація", fontsize=16)
    ax.set_ylabel("$f(x^k) - f^*$", fontsize=16)
    ax.tick_params(axis='both', labelsize=14)
    # plt.xlabel("Ітерація")
    # plt.ylabel("$f(x^k) - f^*$")
    plt.title(f"{en_to_ua_mapping[method_name]} на функції {func['name']}", fontsize=16)
    plt.legend(title=header, loc="upper right", frameon=True, framealpha=0.5)
    plt.grid(True)
    plt.tight_layout()

    plot_path = os.path.join(save_dir, f"{method_name}_{func['name'].replace(' ', '_')}.png")
    plt.savefig(plot_path, dpi=75)
    plt.close()

    print(f"✅ Saved plot: {plot_path}")

    # # Check if the function is 2D for contour plotting
    # if len(func["x_start"]) == 2:
    #
    #     # Plot optimization path
    #     variables = list(sym.ordered(func["function"].free_symbols))
    #     func_numeric = sym.lambdify(variables, func["function"], modules="numpy")
    #
    #     for use_restart in [False, True]:
    #         with_restart_text = "restart" if use_restart else "no_restart"
    #
    #         path_plot_path = os.path.join(save_dir, f"{method_name}_{func['name'].replace(' ', '_')}_{with_restart_text}_trajectory.png")
    #
    #         plot_minimization_paths_2d(
    #             func_numeric,
    #             result_no_restart=results[0],
    #             result_restart=results[1],
    #             x_bounds=func["x_bounds"],
    #             y_bounds=func["y_bounds"],
    #             save_path=path_plot_path,
    #             plot_restart=use_restart,
    #             func_name=func["name"]
    #         )

    # Return result summaries for CSV export
    summary_rows = []
    for res in results:
        x_error = float(np.linalg.norm(res.x_min_true - res.x_min_pred))
        f_error = abs(res.f_min_true - res.f_min_pred)

        summary_rows.append({
            "function_name": func["name"],
            "method_name": f"{res.method_name} {'(restart)' if res.use_restart else '(no restart)'}",
            "x_error": x_error,
            "f_error": f_error,
            "iterations": len(res.errors),
            "has_failed": res.is_denominator_0,
            "dimension": len(res.x_start)
        })

    return summary_rows



def run_all_pairs(precision=1e-4, csv_path="results_summary.csv"):
    functions = generate_test_functions()
    methods = ["cgd", "broyden", "dfp"]  # "broyden",
    all_results = []

    for func in functions:
        for method in methods:
            print(f"\n📌 Running {method.upper()} on {func['name']}")
            result_rows = run_method_pair_on_function(func, method, precision)
            all_results.extend(result_rows)

    # Save to CSV
    df = pd.DataFrame(all_results)
    df.to_csv(csv_path, index=False)
    print(f"\n📄 Results saved to {csv_path}")


if __name__ == "__main__":
    run_all_pairs(1e-6)


# if __name__ == "__main__":
#     run_all_experiments()


"""
Test functions for Broyden
1. Dixon-Price (n=2)
2. Easom
3. Michalewicz (n=2)
4. Perm (n=2, n=5, n=10)
5. Powell (n=4, n=8, n=12)
6. Rosenbrock (n=2, n=5, n=10)

Test functions for CGD
1. Dixon-Price (n=2, n=5, n=10)
2. Easom
3. Michalewicz (n=2)
4. Perm (n=2, n=5, n=10)
5. Powell (n=4, n=8, n=12)
6. Rosenbrock (n=2, n=5, n=10)
7. Six-Hump Camel
8. Zakharov (n=2, n=5, n=10)
9. Golden-Price
"""

