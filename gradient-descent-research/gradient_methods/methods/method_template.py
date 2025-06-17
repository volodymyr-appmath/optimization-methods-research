from abc import ABC, abstractmethod
from typing import List
from textwrap import dedent

import numpy as np
import sympy
import sympy as sym

from numpy.linalg import norm

from gradient_methods.methods.abstract_methods import GradientMethod
from gradient_methods.methods.broyden import Broyden
from gradient_methods.methods.cgd import CGD
from gradient_methods.methods.functional import line_search


class Newton(GradientMethod):
    def __init__(self, function, x_start, precision, use_restart=False, method_name="Newton"):
        super().__init__(function, x_start, precision, method_name)
        self.use_restart = use_restart
        self.n_iterations = 0
        self.x_k = [sym.Matrix(x_start)]  # steps of minimization
        self.s_k = [sym.Matrix(-self.grad(x_start))]  # steps of anti-gradient

if __name__ == "__main__":
    x1, x2 = sym.symbols('x1 x2')
    f_rosen = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2  # True minimum at (1, 1)
    x_0 = [-1.2, 1]
    precision = 0.0001


    cgd_with_restart = CGD(f_rosen, x_0, f_min=0, precision=precision, use_restart=True)
    cgd_without_restart = CGD(f_rosen, x_0, f_min=0, precision=precision, use_restart=False)

    # cgd_without_restart.optimize()


    broyden = Broyden(f_rosen, x_0, f_min=0, precision=precision, use_restart=True, method_name="Broyden")
    # broyden.optimize()

    import matplotlib.pyplot as plt
    from gradient_methods.visualizations import plot_3d


    def plot_log_errors(results: List[np.ndarray]):
        """
        Plots log10 errors.

        :param results: An array containing results of multiple runs.
        :return: None
        """
        plt.figure(figsize=(10, 6))
        for res in results:
            plt.plot(res["errors"], label=res["name"])
        plt.xlabel("Iteration")
        plt.ylabel("log₁₀(f(xₖ) - f_min)")
        plt.title("Convergence Comparison")
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.show()


    def run_all_methods():

        x1, x2 = sym.symbols('x1 x2')

        results = []

        # Quadratic
        f_quad = (x1 - 1) ** 2 + (x2 + 2) ** 2
        x_start_quad = [0.0, 0.0]
        f_min_quad = 0.0

        # broyden_q = Broiden(f_quad, x_start_quad, 1e-6, f_min_quad, use_restart=True)
        # broyden_q.optimize()

        cgd_q = CGD(f_quad, x_start_quad, 1e-6, f_min_quad, line_search_precision=1e-4, use_restart=False)
        cgd_q.optimize()

        # results.append({"name": "Broyden (Quadratic)", "errors": broyden_q.errors})
        # results.append({"name": "CGD (Quadratic)", "errors": cgd_q.errors})

        # Rosenbrock
        f_rosen = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
        x_start_rosen = [-1.2, 1.0]
        f_min_rosen = 0.0

        # broyden_r = Broiden(f_rosen, x_start_rosen, 1e-6, f_min_rosen, use_restart=True)
        # broyden_r.optimize()
        #
        # # broyden_r_without_restart = Broiden(f_rosen, x_start_rosen, 1e-6, f_min_rosen, use_restart=False)
        # # broyden_r_without_restart.optimize()
        #
        # cgd_r = CGD(f_rosen, x_start_rosen, 1e-6, f_min_rosen, use_restart=True)
        # cgd_r.optimize()
        #
        # results.append({"name": "Broyden (Rosenbrock)", "errors": broyden_r.errors})
        # # results.append({"name": "Broyden (w/o restart (Rosenbrock)", "errors": broyden_r_without_restart.errors})
        # results.append({"name": "CGD (Rosenbrock)", "errors": cgd_r.errors})

        plot_log_errors(results)

        # plot_3d(broyden_r.function_symbolic, history=broyden_r.history, xlim_=(-2, 2), ylim_=(-2, 2))
        # plot_3d(broyden_q.function_symbolic, history=broyden_q.history, xlim_=(-4, 4), ylim_=(-6, 4))
        # plot_3d(cgd_q.function_symbolic, history=cgd_q.history, xlim_=(-4, 4), ylim_=(-6, 4))



    run_all_methods()

    # import matplotlib.pyplot as plt
    #
    # plt.plot(range(1, len(broiden.errors) + 1), broiden.errors)
    # plt.xlabel("Iteration")
    # plt.ylabel("log10(f(x_k) - f_min)")
    # plt.title("Convergence of Broyden's Method")
    # plt.grid(True)
    # plt.show()

    # def test_broyden_on_quadratic():
    #     # Define symbolic variables
    #     x1, x2 = sym.symbols('x1 x2')
    #     f_sym = (x1 - 1) ** 2 + (x2 + 2) ** 2
    #     x_start = [0.0, 0.0]
    #     f_min = 0.0
    #
    #     broyden = Broiden(function=f_sym, x_start=x_start, precision=1e-6, f_min=f_min, use_restart=True)
    #     broyden.optimize()
    #
    #     x_opt = broyden.x_k[-1]
    #     f_val = broyden.f(x_opt)
    #
    #     print("\n✅ Quadratic test:")
    #     print("x* =", x_opt)
    #     print("f(x*) =", f_val)
    #
    #     assert np.allclose(x_opt, [1, -2], atol=1e-4), "Did not converge to the correct minimum"
    #     assert abs(f_val - f_min) < 1e-8, "Final value is not accurate"
    #
    #
    # def test_broyden_on_rosenbrock():
    #     x1, x2 = sym.symbols('x1 x2')
    #     f_sym = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
    #     x_start = [-1.2, 1.0]  # Classic Rosenbrock starting point
    #     f_min = 0.0
    #
    #     broyden = Broiden(function=f_sym, x_start=x_start, precision=1e-6, f_min=f_min, use_restart=True)
    #     broyden.optimize()
    #
    #     x_opt = broyden.x_k[-1]
    #     f_val = broyden.f(x_opt)
    #
    #     print("\n✅ Rosenbrock test:")
    #     print("x* =", x_opt)
    #     print("f(x*) =", f_val)
    #
    #     assert norm(x_opt - np.array([1.0, 1.0])) < 1e-2, "Did not converge close enough to the true minimum"
    #     assert abs(f_val - f_min) < 1e-6, "Final value is not accurate"
    #
    #
    # # Run the tests
    # test_broyden_on_quadratic()
    # test_broyden_on_rosenbrock()

    # import matplotlib.pyplot as plt
    #
    #
    # def plot_log_errors(results):
    #     plt.figure(figsize=(8, 5))
    #     for res in results:
    #         plt.plot(res["errors"], label=res["name"])
    #     plt.xlabel("Iteration")
    #     plt.ylabel("log₁₀(f(xₖ) - f_min)")
    #     plt.title("Convergence Comparison")
    #     plt.grid(True)
    #     plt.legend()
    #     plt.tight_layout()
    #     plt.show()
    #
    # def run_all_tests():
    #     import sympy as sym
    #
    #     # Quadratic bowl
    #     x1, x2 = sym.symbols('x1 x2')
    #     f_quad = (x1 - 1) ** 2 + (x2 + 2) ** 2
    #     x_start_quad = [0.0, 0.0]
    #     f_min_quad = 0.0
    #
    #     broyden_q = Broiden(f_quad, x_start_quad, 1e-6, f_min_quad, use_restart=True)
    #     broyden_q.optimize()
    #
    #     # Rosenbrock
    #     f_rosen = 100 * (x2 - x1 ** 2) ** 2 + (1 - x1) ** 2
    #     x_start_rosen = [-1.2, 1.0]
    #
    #     broyden_r = Broiden(f_rosen, x_start_rosen, 1e-6, 0.0, use_restart=True)
    #     broyden_r.optimize()
    #
    #     results = [
    #         {"name": "Broyden (Quadratic)", "errors": broyden_q.errors},
    #         {"name": "Broyden (Rosenbrock)", "errors": broyden_r.errors}
    #     ]
    #
    #     plot_log_errors(results)


    # Run all tests and show plot
    # run_all_tests()








    # cgd_with_restart.optimize()
    # cgd_without_restart.optimize()
    #
    # history_rosen_restart = [cgd_with_restart.history, cgd_with_restart.n_iterations]
    # history_rosen_default = [cgd_without_restart.history, cgd_without_restart.n_iterations]
    # print("------------ Test 1 ------------")
    # print(f"Found point of minimum x_min (without restart): \n{history_rosen_default[0][-1]}")
    # print(f"Found point of minimum x_min (with restart): \n{history_rosen_restart[0][-1]}")
    # print(f"Total iterations (without restart): {history_rosen_default[1]}")
    # print(f"Total iterations (with restart): {history_rosen_restart[1]}")
    # print(f"True point of minimum: (1, 1)\n")
