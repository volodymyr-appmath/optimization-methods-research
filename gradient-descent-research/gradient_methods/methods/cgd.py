from typing import List
from textwrap import dedent

import numpy as np
import sympy as sym

from numpy.linalg import norm

from .abstract_methods import GradientMethod
from .functional import line_search, line_search_smart, golden_section_line_search, line_search_decay


class CGD(GradientMethod):
    """
    Conjugate Gradient Method (CGD) for minimization problems.
    """

    def __init__(
            self,
            function: sym.Function,
            x_start: List[float],
            precision: float,
            x_min: List[float],
            f_min: float,
            use_restart=False,
            line_search_precision=1e-4,
            method_name="CGD"
    ):
        super().__init__(function, x_start, precision, method_name)

        x_start = np.array(x_start)
        self.x_min = np.array(x_min)

        self.use_restart = use_restart
        self.line_search_precision = line_search_precision

        # Store log10 errors
        self.f_min = float(f_min)
        self.errors = []

        # Initial parameters
        self.x_k = [x_start]
        self.s_k = -self.grad(x_start)

    def step(self):
        if self.n_iterations == 0:
            self.n_iterations = 1
        k = self.n_iterations

        # a = sym.Symbol('a')
        f = self.f
        grad = self.grad
        x_k = self.x_k
        s_k = self.s_k

        # Step 1: Choose minimization step using Armijo's line search
        # a = line_search(f, grad, x_k[k - 1], s_k, c=self.line_search_precision, decay=0.5)
        a = line_search(f, grad, x_k[k - 1], s_k.flatten(), c=self.line_search_precision)

        # a = golden_section_line_search(f, x_k[k - 1], s_k.flatten())
        # a = line_search_decay(f, grad, x_k[k - 1], s_k.flatten(), C=1.0)

        # Step 2: Update current point x_k
        x = x_k[k - 1] + a * s_k.flatten()
        x_k.append(x)
        self.x_opt = x

        # Step 3: Calculating beta for the next (k+1) iteration
        grad_cur = grad(x_k[k])
        grad_prev = grad(x_k[k - 1])

        numerator = norm(grad_cur) ** 2
        denominator = norm(grad_prev) ** 2

        # Step 4: Restart procedure: Reset beta parameter to 0 if the number of conjugate directions is equal to dim
        if self.use_restart and k % self.dim == 0:  # k % self.dim == 0:  k % 4 == 0:
            b = 0
            message = f'RESTART MADE ON k = {k}'
        else:
            b = numerator / denominator
            message = ''

        # threshold = 0.9
        # grad_k = grad(x_k[k]).flatten()
        # grad_k_minus_1 = grad(x_k[k - 1]).flatten()
        # cos_angle = abs(np.dot(grad_k, grad_k_minus_1)) / (norm(grad_k) * norm(grad_k_minus_1))
        #
        # if self.use_restart and cos_angle > threshold:  # e.g., 0.8 or 0.9
        #     b = 0
        #     message = f'RESTART MADE ON k = {k}'
        # else:
        #     b = numerator / denominator
        #     message = ''

        # Step 5: Calculate s for the next (k+1) iteration
        self.s_k = -grad_cur + b * s_k

        # --------------- Non-algorithm related ---------------

        # Compute and log error
        f_k = f(x)
        err = abs(f_k - self.f_min)
        err = max(err, 1e-15)  # prevent log10(0) or negative
        log_error = np.log10(err)
        self.errors.append(log_error)

        # Show current optimization state
        print(f'\n{self.current_state(x, a, b, message)}')

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

            if (A1 and A2 and A3) or k >= 1000:
                self.history = self.x_k
                break

            # if norm(x_k[-1] - self.x_min) <= self.precision or k >= 1000:
            # if abs(f(x_k[-1]) - self.f_min) <= self.precision or k >= 1000:
            #     self.history = self.x_k
            #     break

            # threshold = 0.9
            # grad_k = self.grad(x_k[k])
            # grad_k_minus_1 = self.grad(x_k[k-1])
            # cos_angle = abs(np.dot(grad_k.T, grad_k_minus_1)) / (norm(grad_k) * norm(grad_k_minus_1))
            # if cos_angle > threshold:  # e.g., 0.8 or 0.9
            #     beta = 0

            self.n_iterations += 1

    def current_state(self, x_cur: np.ndarray, alpha_cur: float, beta_cur: float, restart_message: str):
        state = dedent(f"""
            --- Iteration: {self.n_iterations} ---
            x_{self.n_iterations}: {x_cur}
            alpha_{self.n_iterations} = {alpha_cur}
            beta_{self.n_iterations} = {beta_cur}
            log10(f(x_{self.n_iterations}) - f_min) = {self.errors[-1]}
            {restart_message}
        """)
        return state
