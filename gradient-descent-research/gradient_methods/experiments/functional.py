from typing import List

from matplotlib import pyplot as plt

from .constants import method_name_mapping, en_to_ua_mapping
from .data_objects import MinimizationResults, ExperimentConfig


def run_experiment(
        experiment_config: ExperimentConfig,
):
    """
    Runs a single experiment. An experiment includes:
    - Minimizing the given function;
    - Calculating errors with respect to the number of iterations;
    - Gathering results.

    :param experiment_config: Configuration of the experiment (function, opt method, x_start, precision, etc.).
    :return: Results of the minimization process.
    """

    function = experiment_config.function
    x_start = experiment_config.x_start
    x_min_true = experiment_config.x_min_true
    f_min_true = experiment_config.f_min_true
    precision = experiment_config.precision
    line_search_precision = experiment_config.line_search_precision
    use_restart = experiment_config.use_restart
    method_name = experiment_config.method_name

    method = method_name_mapping[experiment_config.method_name.lower()](
        function=function,
        x_start=x_start,
        x_min=x_min_true,
        precision=precision,
        f_min=f_min_true,
        use_restart=use_restart,
        line_search_precision=line_search_precision,
    )

    method.optimize()

    x_min_pred = method.x_k[-1]
    f_min_pred = method.f(x_min_pred)
    n_iterations = method.n_iterations
    errors = method.errors

    # label = method_name
    # if use_restart:
    #     label += " (restart)"

    results_dict = {
        "function": function,
        "x_start": x_start,
        "x_min_true": x_min_true,
        "x_min_pred": x_min_pred,
        "f_min_true": f_min_true,
        "f_min_pred": f_min_pred,
        "n_iterations": n_iterations,
        "errors": errors,
        "method_name": en_to_ua_mapping[method_name],
        "use_restart": use_restart,
        "precision": precision,
        "history": method.history,
        "is_denominator_0": method.is_denominator_0 if hasattr(method, 'is_denominator_0') else False,
    }

    results = MinimizationResults(**results_dict)

    return results


def plot_log_errors(results: List[MinimizationResults]):
    """
    Plots log10 errors.

    :param results: An array containing results of multiple runs.
    :return: None
    """
    plt.figure(figsize=(10, 6))
    for res in results:
        plt.plot(res.errors, label=res.method_name, linestyle='--')
    plt.xlabel("Iteration")
    plt.ylabel("log₁₀(f(xₖ) - f_min)")
    plt.title("Convergence Comparison")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
