from typing import Callable

import numpy as np
import sympy as sym
from sympy import solve


def line_search(f: Callable, grad_f: Callable, x: np.ndarray, d: np.ndarray, decay=0.5, c=1e-4):
    """
    Armijo's line search algorithm.

    :param x: point x == [x1, x2, ..., xn]
    :param f: Callable function (so that f(x) can be computed).
    :param grad_f: Callable gradient (so that gradient(x) can be computed).
    :param d: The product of the Hessian's inverse and the gradient.
    :param decay: The decay factor for beta.
    :param c: Precision constant.
    :return:
    """
    beta = 0.999

    arg = x + beta * d.flatten()

    # print(f(arg), f(x) - c * beta * np.dot(grad_f(x).flatten(), d.flatten()))
    while f(arg) > f(x) - c * beta * np.dot(grad_f(x).flatten(), d.flatten()):
        # print(f(arg), f(x) - c * beta * np.dot(grad_f(x).flatten(), d.flatten()))
        beta *= decay
        arg = x + beta * d.flatten()

    return beta


def line_search_smart(f, grad_f, x, d, c=1e-4, beta_init=0.999, decay=0.5, delta=0.05, max_iter=200):
    """
    Smart line search:
    1. Shrinks β geometrically until Armijo condition is met.
    2. Then grows β linearly until Armijo fails again.
    Returns the largest valid β found.
    """
    beta = beta_init
    fx = f(x)
    grad_dot_d = np.dot(grad_f(x).flatten(), d.flatten())

    # Phase 1: Backtrack until Armijo condition is met
    for _ in range(max_iter):
        x_new = x + beta * d
        if f(x_new) <= fx - c * beta * grad_dot_d:
            break
        beta *= decay
    else:
        print(f"⚠️ Smart line search: no valid β found. Using β = {beta}")
        return beta

    # Phase 2: Try increasing β by delta
    best_beta = beta
    for _ in range(max_iter):
        candidate = best_beta + delta
        x_try = x + candidate * d
        if f(x_try) <= fx - c * candidate * grad_dot_d:
            best_beta = candidate
        else:
            break

    return best_beta


def line_search_decay(f, grad_f, x, d, c=1e-4, C=0.5, beta_init=0.999, max_iter=200):
    """
    Armijo line search with adaptive decay: beta *= C / (k + 1)
    """
    beta = beta_init
    fx = f(x)
    grad_dot_d = np.dot(grad_f(x).flatten(), d.flatten())

    for k in range(max_iter):
        x_new = x + beta * d
        f_new = f(x_new)

        if f_new <= fx - c * beta * grad_dot_d:
            return beta

        # Adaptive shrinkage
        beta *= C / (k + 1)

    print("⚠️ Adaptive line search failed. Falling back to α = 1.0")
    return 1.0


from scipy.optimize import minimize_scalar
import numpy as np


def bracket_minimum(f, x, d, alpha_init=1.0, growth=2.0, max_iter=10):
    """
    Expands [0, b] until f(x + b*d) > f(x + a*d), assuming unimodal function.
    """
    a = 0.0
    b = alpha_init
    fa = f(x + a * d)
    fb = f(x + b * d)
    it = 0
    while fb < fa and it < max_iter:
        a = b
        fa = fb
        b = b * growth
        fb = f(x + b * d)
        it += 1
    return 0.0, b


def golden_section_line_search(f, x, d):
    """
    Uses golden section search to minimize f(x + α * d).

    :param f: function f(x)
    :param x: current point (np.ndarray)
    :param d: descent direction (np.ndarray)
    :return: optimal α along direction d
    """
    phi = lambda alpha: f(x + alpha * d)
    a, b = bracket_minimum(f, x, d)

    result = minimize_scalar(phi, method="golden", bracket=(a, b), options={"maxiter": 200})
    return result.x if result.success else 1.0  # fallback


def substitute(func, variables, expression):
    """
    Assisting function to shorten the operation of passing an expression to
    the input function and its gradient.

    :param func: the given function
    :param variables: the variables to substitute
    :param expression: function's argument

    :return: function's value at a given point
    """
    result = sym.N(func.subs(list(zip(variables, expression))))
    return result


def norm(vector):
    """
    Calculates norm of the given vector

    :param vector:
    :return:
    """
    return np.linalg.norm(np.array(sym.N(vector)).astype(float))


def minimize_1d(func, variable):
    """
    1d minimization of the given function (performed by sympy.minimum())

    :param func: given function
    :param variable: variable which is to be minimized by

    :return: point of global minimum
    """
    f_min = sym.N(sym.minimum(func, variable))  # returns minimum of a function
    x_min = [sym.re(num) for num in solve(func - f_min, variable)]  # choosing only real parts if roots are complex
    x_min = sym.N(min(list(map(abs, list(map(float, x_min))))))  # converting precise value into float
    return x_min
