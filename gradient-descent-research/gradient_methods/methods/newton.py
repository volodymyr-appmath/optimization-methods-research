from typing import List
from textwrap import dedent

import numpy as np
import sympy as sym

from numpy.linalg import norm, solve

from .abstract_methods import GradientMethod
from .functional import line_search, line_search_smart


class NewtonMethod(GradientMethod):
    """
    Damped Newton's method for unconstrained minimization.
    """

    def __init__(
            self,
            function: sym.Function,
            x_start: List[float],
            precision: float,
            x_min: List[float],
            f_min: float,
            use_restart=False,
            line_search_precision: float = 1e-4,
            restart_precision=1e6,
            method_name: str = "Newton"
    ):
        super().__init__(function, x_start, precision, method_name)

        self.use_restart = use_restart
        self.line_search_precision = line_search_precision
        self.restart_precision = restart_precision

        # Store log10 errors
        self.f_min = float(f_min)
        self.errors = []

        # Initial parameters
        x_start = np.array(x_start)
        self.x_k = [x_start]

        # Symbolic Hessian
        self.hessian_symbolic = sym.hessian(self.function_symbolic, self.variables)
        self.hessian_numeric = sym.lambdify(self.variables, self.hessian_symbolic, modules='numpy')

    def hess(self, x: np.ndarray) -> np.ndarray:
        return np.array(self.hessian_numeric(*x)).reshape(self.dim, self.dim)

    def step(self):
        if self.n_iterations == 0:
            self.n_iterations = 1
        k = self.n_iterations

        f = self.f
        grad = self.grad
        hess = self.hess
        x_k = self.x_k

        grad_k = grad(x_k[k - 1])
        hess_k = hess(x_k[k - 1])

        # Direction: d = -H^{-1} * grad
        # det_hess = abs(np.linalg.det(hess_k))
        cond_hess = np.linalg.cond(hess_k)  # Condition number of the Hessian

        if k != 1 and self.use_restart and (cond_hess >= self.restart_precision or np.isnan(cond_hess)):  # abs(det_hess <= 1e-8):
            message = f"RESTART on iteration {k}: ill-conditioned Hessian (cond={cond_hess:.2e})"
            d_k = -grad_k.flatten()
        else:
            d_k = -solve(hess_k, grad_k).flatten()
            message = ''

        alpha = line_search(f, grad, x_k[k - 1], d_k, c=self.line_search_precision)

        # Update step
        x = x_k[k - 1] + alpha * d_k
        x_k.append(x)

        # Compute and log error
        f_k = f(x)
        err = abs(f_k - self.f_min)
        err = max(err, 1e-15)
        self.errors.append(np.log10(err))

        print(f"\n{self.current_state(x, alpha, message)}")

    def optimize(self):
        while True:
            self.step()

            k = self.n_iterations
            f = self.f
            grad = self.grad
            x_k = self.x_k

            A1 = abs(f(x_k[k - 1]) - f(x_k[k])) < self.precision * (1 + abs(f(x_k[k])))
            A2 = norm(x_k[k - 1] - x_k[k]) < np.sqrt(self.precision) * (1 + norm(x_k[k]))
            A3 = norm(grad(x_k[k])) <= self.precision ** (1 / 3) * (1 + abs(f(x_k[k])))

            if (A1 and A2 and A3) or k >= 1000:
                self.history = self.x_k
                self.x_opt = self.x_k[-1]
                break

            self.n_iterations += 1

    def current_state(self, x_cur: np.ndarray, alpha: float, restart_message: str):
        state = dedent(f"""
            --- Iteration: {self.n_iterations} ---
            x_{self.n_iterations}: {x_cur}
            alpha_{self.n_iterations} = {alpha}
            log10(f(x_{self.n_iterations}) - f_min) = {self.errors[-1]}
            {restart_message}
        """)
        return state
