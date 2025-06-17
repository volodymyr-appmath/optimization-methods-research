from typing import List
from textwrap import dedent

import numpy as np
import sympy as sym

from numpy.linalg import norm

from .abstract_methods import GradientMethod
from .functional import line_search


class DFP(GradientMethod):
    """
    Davidon-Fletcher-Powell (DFP) method for unconstrained minimization.
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
            method_name="DFP"
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
        d = (-H @ grad_x_prev).flatten()

        # Step 2: Choose minimization step
        b = line_search(f, grad, x_k[k - 1], d, c=self.line_search_precision)

        # Step 3: Update x
        x = x_k[k - 1] + b * d
        x_k.append(x)
        self.x_opt = x

        # Step 4: Update inverse Hessian using DFP formula
        grad_x_cur = grad(x_k[k])
        dy = grad_x_cur - grad_x_prev  # (n, 1)
        dx = (x_k[k] - x_k[k - 1]).reshape(self.dim, 1)

        dy = dy.reshape(self.dim, 1)

        Hdy = H @ dy
        dx_dy = float(dx.T @ dy)
        dy_Hdy = float(dy.T @ Hdy)

        if self.use_restart and abs(dx_dy*dy_Hdy) <= self.restart_precision:
            self.H = np.identity(self.dim)
            message = f'RESTART MADE ON k = {k}, dx_dy = {dx_dy}'
        else:
            if abs(dx_dy*dy_Hdy) <= self.restart_precision:
                self.is_denominator_0 = True
            else:
                term1 = (dx @ dx.T) / dx_dy
                term2 = (Hdy @ Hdy.T) / dy_Hdy
                self.H = H + term1 - term2
            message = ''

        # Log error
        f_k = f(x)
        err = abs(f_k - self.f_min)
        err = max(err, 1e-15)
        log_error = np.log10(err)
        self.errors.append(float(log_error))

        print(f'\n{self.current_state(x, b, message)}')

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
