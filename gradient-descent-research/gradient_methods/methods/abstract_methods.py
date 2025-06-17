from abc import ABC, abstractmethod
from typing import List
from textwrap import dedent

import numpy as np
import sympy as sym


class OptMethod(ABC):
    """
    Abstract class for optimization methods.
    """
    def __init__(
            self,
            function: sym.Function,
            x_start: List[float],
            precision: float,
            method_name="AbstractMethod"
    ):
        self.function_symbolic = function

        self.x_start = x_start
        self.x_opt = x_start.copy()

        self.precision = precision
        self.method_name = method_name

        self.variables = list(sym.ordered(function.free_symbols))
        self.dim = len(self.variables)
        self.function_numeric = sym.lambdify(self.variables, self.function_symbolic, modules='numpy')

        self.n_iterations = 0
        self.history = []  # Stores points at each minimization step

        assert self.dim == len(x_start)

    @abstractmethod
    def step(self):
        """
        Performs one optimization step.
        """
        print("Not implemented")

    @abstractmethod
    def optimize(self):
        """
        Performs multiple optimization steps.
        """
        print("Not implemented")

    @abstractmethod
    def current_state(self, **kwargs):
        """
        Returns the current state of the optimization.
        """
        print("Not implemented")

    def f(self, x: np.ndarray) -> float:
        """
        Computes the function at the given point.
        The number of variables in x should be equal to those in the function.
        :param x: point x == [x1, x2, ..., xn]
        :return: f(x) == f(x1, x2, ..., xn)
        """
        assert x.shape == (self.dim, )  # x should be a flat array
        return self.function_numeric(*x)

    def reset(self):
        self.x_opt = self.x_start.copy()
        self.history = []
        self.n_iterations = 0

    def __call__(self, x):
        return self.f(x)

    def __repr__(self):
        header = f"{'='*20} {self.method_name} {'='*20}"

        description = dedent(f"""
            {header}
            f = {self.function_symbolic}
            dim = {self.dim}
            x_start = {self.x_start}
            x_opt = {self.x_opt}
            precision = {self.precision}
            {"=" * len(header)}
        """)

        return description


class GradientMethod(OptMethod, ABC):
    """
    Base class for gradient-based optimization methods.
    """
    def __init__(
            self,
            function: sym.Function,
            x_start: List[float],
            precision: float,
            method_name="AbstractMethod"
    ):
        super().__init__(function, x_start, precision, method_name)

        self.gradient_symbolic = [self.function_symbolic.diff(x) for x in self.variables]
        self.gradient_numeric = sym.lambdify(self.variables, self.gradient_symbolic, modules='numpy')

    def grad(self, x: np.ndarray) -> np.ndarray:
        """
        Computes the gradient at the given point.
        The number of variables in x should be equal to those in the original function.

        :param x: point x == [x1, x2, ..., xn]
        :return: gradient(x) == [df/dx1(x1), df/dx2(x2), ..., df/dxn(xn)]
        """
        return np.array(self.gradient_numeric(*x)).reshape(self.dim, 1)
