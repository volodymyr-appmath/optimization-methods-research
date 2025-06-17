# from typing import List
# from textwrap import dedent
#
# import numpy as np
# import sympy as sym
#
# from numpy.linalg import norm
#
# from .abstract_methods import GradientMethod
# from .functional import line_search


# class Broiden(GradientMethod):
#     def __init__(self, function, x_start, precision, use_restart=False, restart_precision=1e-6, method_name="Broiden"):
#         super().__init__(function, x_start, precision, method_name)
#         self.use_restart = use_restart
#         self.restart_precision = restart_precision
#         self.n_iterations = 0
#         self.x_k = [np.array(x_start)]  # steps of minimization
#         self.H = np.identity(self.dim)   # H0 Is an identity matrix
#
#     def step(self):
#         if self.n_iterations == 0:
#             self.n_iterations = 1
#         k = self.n_iterations
#
#         f = self.f
#         grad = self.grad
#         x_k = self.x_k
#         H = self.H
#
#
#         print(f'\n\n----------- {k} -----------\n\n')
#
#         # Step 1: Calculate minimization direction
#         grad_x_prev = np.array(grad(x_k[k - 1])).astype(float)
#         d = (-H @ grad_x_prev).flatten()  # (n, n) x (n, 1) = (n, 1) --> (1, n)
#
#         # Step 2: Choose minimization step
#         b = choose_beta(x_k[k - 1], f, grad, d)
#
#         # Step 3: Update current point x_k
#         x = x_k[k - 1] + b * d
#         x_k.append(x)
#         print(f'\nx_start: {self.x_start}, x_{k}: {x}')
#
#         # Step 4: Update the inverse Hessian H_k using Broiden's method
#         grad_x_cur = np.array(grad(x_k[k])).astype(float)
#         dy = grad_x_cur - grad_x_prev
#         dx = (x_k[k] - x_k[k - 1]).reshape(self.dim, 1)
#
#         z = dx - H @ dy  # (n, 1) - (n, n) x (n, 1) = (n, 1) - (n, 1) = (n, 1)
#         dot_product = np.dot(z.flatten(), dy.flatten())
#
#         # Restart procedure: Reset H_k to identity matrix if the dot_product is approaching 0
#         if self.use_restart and abs(dot_product) <= self.restart_precision: # оновлення
#             # if k % self.dim == 0:
#             self.H = np.identity(self.dim)
#             print(f'\nRESTART MADE ON k = {k}, dot_product = {dot_product}')
#         else:
#             self.H = H + (z @ z.T) / dot_product  # (n, n) + (n, 1) x (1, n) / (,1) = (n, n) + (n, n) = (n, n)
#
#         # print(f'\ndot_product z * dy: {dot_product}\n')


# class Broiden(GradientMethod):
#     def __init__(self, function, x_start, precision, f_min, use_restart=False, restart_precision=1e-6,
#                  method_name="Broiden"):
#         super().__init__(function, x_start, precision, method_name)
#         self.use_restart = use_restart
#         self.restart_precision = restart_precision
#         self.f_min = float(f_min)
#         self.errors = []  # Store log10 errors
#         self.n_iterations = 0
#         self.x_k = [np.array(x_start)]
#         self.H = np.identity(self.dim)
#
#     def step(self):
#         if self.n_iterations == 0:
#             self.n_iterations = 1
#         k = self.n_iterations
#
#         f = self.f
#         grad = self.grad
#         x_k = self.x_k
#         H = self.H
#
#         print(f'\n\n----------- {k} -----------\n\n')
#
#         # Step 1: Compute direction
#         grad_x_prev = np.array(grad(x_k[k - 1])).astype(float)
#         d = (-H @ grad_x_prev).flatten()
#
#         # Step 2: Line search
#         b = choose_beta(x_k[k - 1], f, grad, d)
#
#         # Step 3: Update x
#         x = x_k[k - 1] + b * d
#         x_k.append(x)
#
#         print(f'\nx_start: {self.x_start}, x_{k}: {x}')
#
#         # Step 4: Broyden update
#         grad_x_cur = np.array(grad(x_k[k])).astype(float)
#         dy = grad_x_cur - grad_x_prev
#         dx = (x_k[k] - x_k[k - 1]).reshape(self.dim, 1)
#
#         z = dx - H @ dy
#         dot_product = np.dot(z.flatten(), dy.flatten())
#
#         # Step 5: Restart or update H
#         if self.use_restart and abs(dot_product) <= self.restart_precision:
#             self.H = np.identity(self.dim)
#             print(f'\nRESTART MADE ON k = {k}, dot_product = {dot_product}')
#         else:
#             self.H = H + (z @ z.T) / dot_product
#
#         # Step 6: Compute and log error
#         f_k = float(f(sym.Matrix(x.tolist())))
#         err = f_k - self.f_min
#         err = max(err, 1e-15)  # prevent log10(0) or negative
#         log_error = np.log10(err)
#         self.errors.append(log_error)
#         print(f"log10(f(x_{k}) - f_min) = {log_error}")



# class CGD(GradientMethod):
#     def __init__(self, function, x_start, precision, use_restart=False, method_name="CGD"):
#         super().__init__(function, x_start, precision, method_name)
#         self.use_restart = use_restart
#         self.n_iterations = 0
#         self.x_k = [sym.Matrix(x_start)]  # steps of minimization
#         self.s_k = [sym.Matrix(-self.grad(x_start))]  # steps of anti-gradient
#
#
#     def step(self):
#         if self.n_iterations == 0:
#             self.n_iterations = 1
#         k = self.n_iterations
#
#         a = sym.Symbol('a')
#         f = self.f
#         grad = self.grad
#         x_k = self.x_k
#         s_k = self.s_k
#
#         # Calculating x_k and f(x_k) for current (k) iteration
#         x = x_k[k - 1] + a * s_k[k - 1]
#         f_a = f(x)
#         alpha = minimize_1d(f_a, a)
#         x_k.append(sym.N(x.subs(a, alpha)))
#
#         # print_iterations()  # prints only if show_iterations=True
#
#         # Calculating beta and s for the next (k+1) iteration
#         numerator = norm(grad(x_k[k])) ** 2
#         denominator = norm(grad(x_k[k - 1])) ** 2
#
#         if self.use_restart:
#             b = numerator / denominator if k % self.dim != 0 else 0  # оновлення
#         else:
#             b = numerator / denominator
#
#         s = -grad(x_k[k]) + b * s_k[k - 1]
#         s_k.append(s)
#
#         self.current_state(b, alpha, f_a, x)
#
#     def optimize(self):
#         while True:
#             self.step()
#
#             k = self.n_iterations
#             f = self.f
#             grad = self.grad
#             x_k = self.x_k
#
#             # Stop criterion (with conditions A1-A3)
#             A1 = abs(sym.N(f(x_k[k-1]) - f(x_k[k]))) < self.precision * (1 + abs(sym.N(f(x_k[k]))))
#             A2 = norm(x_k[k-1] - x_k[k]) < np.sqrt(self.precision) * (1 + norm(x_k[k]))
#             A3 = norm(grad(x_k[k])) <= self.precision**(1/3) * (1 + abs(sym.N(f(x_k[k]))))
#
#             if A1 and A2 and A3:
#                 self.history = self.x_k
#                 break
#
#             self.n_iterations += 1
#
#     def current_state(self, b, alpha, f_a, x):
#         k = self.n_iterations
#         f = self.f
#         grad = self.grad
#         x_k = self.x_k
#         s_k = self.s_k
#
#         print(f"k={k} -----------------------------------")
#         print('\n|Parameters|:')
#         print(f"\t-grad(x_{k - 1}) = {-grad(x_k[k - 1])}\n")
#         print(f"\ts_{k - 1} = {s_k[k - 1]}\n")
#         print(f"\tbeta = {b}\n")
#         print(f"\talpha = {alpha}\n")
#         print('|GD Results|:')
#         print(f"\t(symbolic) x_{k} = {x}\n")
#         print(f"\tf_a = {f_a}\n")
#         print(f"\tx_{k} = {x_k[k]}\n")
#         print(f"A1: {abs(sym.N(f(x_k[k - 1]) - f(x_k[k])))} ? {self.precision * (1 + np.abs(sym.N(f(x_k[k]))))}")
#         print(f"A2: {norm(x_k[k - 1] - x_k[k])} ? {np.sqrt(self.precision) * (1 + norm(x_k[k]))}")
#         print(f"A3: {norm(grad(x_k[k]))} ? {self.precision ** (1 / 3) * (1 + abs(sym.N(f(x_k[k]))))}\n")