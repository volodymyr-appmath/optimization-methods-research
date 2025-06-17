from dataclasses import dataclass
from typing import List

import numpy as np
import sympy as sym


@dataclass
class ExperimentConfig:
    """
    Class serving as a config of the experiment. Stores all input parameters of the minimization problem.

    Params:
        - function -- minimized function.
        - x_start -- starting point of the minimization problem.
        - x_min_true -- true point of minimum.
        - f_min_true -- true minimum f(x_min_true).
        - precision -- precision of minimization.
        - line_search_precision: precision of line search.
        - method_name -- name of the chosen minimization method.
    """

    function: sym.Function
    x_start: np.ndarray
    x_min_true: np.ndarray
    f_min_true: float
    precision: float
    line_search_precision: float
    use_restart: bool
    method_name: str


@dataclass
class MinimizationResults:
    """
    Class for storing minimization results.

    Params:
        - function -- minimized function.
        - x_start -- starting point of the minimization problem.
        - x_min_true -- true point of minimum.
        - x_min_pred -- predicted point of minimum by the chosen method.
        - f_min_true -- true minimum f(x_min_true).
        - f_min_pred -- predicted minimum f(x_min_pred).
        - n_iterations -- number of iterations.
        - errors -- log10 errors for f(xi) - f_min_true.
        - method_name -- name of the chosen minimization method.
    """

    function: sym.Function
    x_start: np.ndarray
    x_min_true: np.ndarray
    x_min_pred: np.ndarray
    f_min_true: float
    f_min_pred: float
    n_iterations: float
    errors: np.ndarray
    method_name: str
    use_restart: bool
    precision: float
    history: List[np.ndarray]
    is_denominator_0: bool
