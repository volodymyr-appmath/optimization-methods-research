from typing import List
from textwrap import dedent

import numpy as np
import sympy as sym

from numpy.linalg import norm

from .abstract_methods import GradientMethod
from .functional import line_search, line_search_smart


class Broyden(GradientMethod):
    """
    Broyden's method for minimization problems.
    """

    def __init__(
            self,
            function: sym.Function,
            x_start: List[float],
            precision: float,
            x_min: float,
            f_min: float,
            use_restart=False,
            line_search_precision=1e-4,
            restart_precision=1e-8,
            method_name="Broyden"
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
        self.H = np.identity(self.dim)
        self.is_denominator_0 = False

    def step(self):
        if self.n_iterations == 0:
            self.n_iterations = 1
        k = self.n_iterations

        f = self.f
        grad = self.grad
        x_k = self.x_k
        H = self.H

        # Step 1: Compute direction
        grad_x_prev = grad(x_k[k - 1])
        d = (-H @ grad_x_prev).flatten()  # (n, n) x (n, 1) = (n, 1) --> (1, n)

        # Step 2: Choose minimization step using Armijo's line search
        # b = line_search(f, grad, x_k[k - 1], d.flatten(), c=self.line_search_precision)
        b = line_search(f, grad, x_k[k - 1], d.flatten(), c=self.line_search_precision)

        # Step 3: Update current point x_k
        x = x_k[k - 1] + b * d
        x_k.append(x)
        self.x_opt = x

        # Step 4: Update the inverse Hessian H_k using Broyden's method for the next (k+1) iteration
        grad_x_cur = grad(x_k[k])
        dy = grad_x_cur - grad_x_prev
        dx = (x_k[k] - x_k[k - 1]).reshape(self.dim, 1)

        z = dx - H @ dy  # (n, 1) - (n, n) x (n, 1) = (n, 1) - (n, 1) = (n, 1)
        dot_product = np.dot(z.flatten(), dy.flatten())

        # Step 5: Restart procedure: Reset H_k to identity matrix if the dot_product is approaching 0
        if self.use_restart and abs(dot_product) <= self.restart_precision:
            self.H = np.identity(self.dim)
            message = f'RESTART MADE ON k = {k}, dot_product = {dot_product}'
        else:
            if abs(dot_product) <= self.restart_precision:
                self.is_denominator_0 = True
            else:
                self.H = H + (z @ z.T) / dot_product  # (n, n) + (n, 1) x (1, n) / (,1) = (n, n) + (n, n) = (n, n)
            message = ''

        # --------------- Non-algorithm related ---------------

        # Compute and log error
        f_k = f(x)
        err = abs(f_k - self.f_min)
        err = max(err, 1e-15)  # prevent log10(0) or negative
        log_error = np.log10(err)
        self.errors.append(float(log_error))

        # Show current optimization state
        print(f'\n{self.current_state(x, b, message)}')

        # -----------------------------------------------------

    def optimize(self):
        while True:
            self.step()

            k = self.n_iterations
            f = self.f
            grad = self.grad
            x_k = self.x_k

            # Stop criterion (with conditions A1-A3)
            A1 = abs(f(x_k[k - 1]) - f(x_k[k])) < self.precision * (1 + abs(f(x_k[k])))
            A2 = norm(x_k[k - 1] - x_k[k]) < np.sqrt(self.precision) * (1 + norm(x_k[k]))
            A3 = norm(grad(x_k[k])) <= self.precision ** (1 / 3) * (1 + abs(f(x_k[k])))

            if (A1 and A2 and A3) or k >= 1000 or self.is_denominator_0:
                self.history = self.x_k
                break

            self.n_iterations += 1

    def current_state(self, x_cur: np.ndarray, beta_cur: float, restart_message: str):
        state = dedent(f"""
            --- Iteration: {self.n_iterations} ---
            x_{self.n_iterations}: {x_cur}
            beta_{self.n_iterations} = {beta_cur}
            log10(f(x_{self.n_iterations}) - f_min) = {self.errors[-1]}
            {restart_message}
        """)
        return state
