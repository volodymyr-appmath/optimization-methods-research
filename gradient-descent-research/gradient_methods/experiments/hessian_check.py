import numpy as np
import sympy as sym

from gradient_methods.methods import NewtonMethod

# Define a test function: Rosenbrock
x1, x2 = sym.symbols('x1 x2')
f_sym = 100 * (x2 - x1**2)**2 + (1 - x1)**2

# Initialize NewtonMethod
method = NewtonMethod(
    function=f_sym,
    x_start=[-1.2, 1],
    precision=1e-8,
    x_min=[1, 1],
    f_min=0,
    use_restart=False
)

# Point to evaluate
x0 = np.array([-1.2, 1.0])

# Get symbolic Hessian from the class
H_sym = method.hess(x0)

# Finite difference approximation of Hessian
def numerical_hessian(f, x, eps=1e-5):
    n = len(x)
    hess = np.zeros((n, n))
    fx = f(x)
    for i in range(n):
        x1 = x.copy()
        x1[i] += eps
        f1 = f(x1)
        for j in range(n):
            x2 = x.copy()
            x2[j] += eps
            if i == j:
                x_ij = x.copy()
                x_ij[i] += eps
                f_ij = f(x_ij)
                hess[i, j] = (f(x + eps*np.eye(n)[i]) - 2 * fx + f(x - eps*np.eye(n)[i])) / eps**2
            else:
                x_pp = x.copy(); x_pp[i] += eps; x_pp[j] += eps
                x_pm = x.copy(); x_pm[i] += eps; x_pm[j] -= eps
                x_mp = x.copy(); x_mp[i] -= eps; x_mp[j] += eps
                x_mm = x.copy(); x_mm[i] -= eps; x_mm[j] -= eps
                hess[i, j] = (f(x_pp) - f(x_pm) - f(x_mp) + f(x_mm)) / (4 * eps**2)
    return hess

H_fd = numerical_hessian(method.f, x0)

# Compare
print("Symbolic Hessian (class):")
print(H_sym)
print("\nNumerical Hessian (finite differences):")
print(H_fd)
print("\nDifference:")
print(H_sym - H_fd)
print("\nMax absolute error:")
print(np.max(np.abs(H_sym - H_fd)))
