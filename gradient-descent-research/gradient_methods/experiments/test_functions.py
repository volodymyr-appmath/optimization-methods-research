import sympy as sym
import numpy as np


def get_plate_shaped_functions():
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = sym.symbols("x1 x2 x3 x4 x5 x6 x7 x8 x9 x10")
    functions = []

    # --- Booth Function (2D) ---
    booth = (x1 + 2*x2 - 7)**2 + (2*x1 + x2 - 5)**2
    functions.append({
        "name": "Booth",
        "function": booth,
        "x_start": np.array([0.0, 0.0]),
        "x_min": np.array([1.0, 3.0]),
        "f_min": 0.0,
        "x_bounds": [-2.0, 7],
        "y_bounds": [-2.0, 7]
    })

    # # --- McCormick Function (2D) ---
    # mccormick = sym.sin(x1 + x2) + (x1 - x2)**2 - 1.5 * x1 + 2.5 * x2 + 1
    # functions.append({
    #     "name": "McCormick",
    #     "function": mccormick,
    #     "x_start": np.array([-1.0, 1.0]),
    #     "x_min": np.array([-0.54719, -1.54719]),  # approx
    #     "f_min": -1.9133  # approx
    # })

    # # --- Matyas Function (2D) ---
    # matyas = 0.26 * (x1**2 + x2**2) - 0.48 * x1 * x2
    # functions.append({
    #     "name": "Matyas",
    #     "function": matyas,
    #     "x_start": np.array([1.0, 1.0]),
    #     "x_min": np.array([0.0, 0.0]),
    #     "f_min": 0.0
    # })

    # --- Zakharov Function (n=2, 5, 10) ---
    for n in [2, 5, 10]:
        xs = sym.symbols(f'x1:{n+1}')  # x1 to xn
        i = np.arange(1, n+1)
        f_expr = sum(x**2 for x in xs)
        f_expr += (sum(0.5 * i[j-1] * xs[j-1] for j in range(1, n+1)))**2
        f_expr += (sum(0.5 * i[j-1] * xs[j-1] for j in range(1, n+1)))**4

        functions.append({
            "name": f"Zakharov (n={n})",
            "function": f_expr,
            "x_start": np.full(n, 0.5),
            "x_min": np.zeros(n),
            "f_min": 0.0,
            "x_bounds": [-1.5, 1.5],
            "y_bounds": [-1.5, 1.5]
        })

    return functions


def generate_rosenbrock_start(n: int):
    x = np.zeros(n)
    x[0] = -1.0
    for i in range(1, n):
        x[i] = x[i - 1]**2 - 0.2
    return x


def get_valley_shaped_functions():
    x1, x2 = sym.symbols("x1 x2")
    functions = []

    # --- Rosenbrock Function (n = 2, 5, 10) ---
    for n in [2, 5, 10]:
        xs = sym.symbols(f'x1:{n+1}')  # x1 to xn
        expr = sum(
            100 * (xs[i+1] - xs[i]**2)**2 + (1 - xs[i])**2
            for i in range(n - 1)
        )
        functions.append({
            "name": f"Rosenbrock (n={n})",
            "function": expr,
            "x_start": generate_rosenbrock_start(n) if n > 2 else np.array([-1.2, 1]), # np.full(n, -1.2),
            "x_min": np.ones(n),
            "f_min": 0.0,
            "x_bounds": [-1.5, 1.3],
            "y_bounds": [-0.75, 1.3]
        })

    # --- Dixon-Price Function (n = 2, 5, 10) ---
    for n in [2, 5, 10]:
        xs = sym.symbols(f'x1:{n+1}')
        expr = (xs[0] - 1)**2 + sum(
            i * (2 * xs[i]**2 - xs[i-1])**2
            for i in range(1, n)
        )
        functions.append({
            "name": f"Dixon-Price (n={n})",
            "function": expr,
            "x_start": np.full(n, 0.5),
            "x_min": np.array([2**(-(2**i - 2)/2**i) for i in range(1, n+1)]),  # approximated
            "f_min": 0.0,
            "x_bounds": [-0, 3],
            "y_bounds": [-0, 1.5]
        })

    # # --- Three-Hump Camel Function (2D) ---
    # camel3 = 2 * x1**2 - 1.05 * x1**4 + (x1**6) / 6 + x1 * x2 + x2**2
    # functions.append({
    #     "name": "Three-Hump Camel",
    #     "function": camel3,
    #     "x_start": np.array([0.5, -0.5]),
    #     "x_min": np.array([0.0, 0.0]),
    #     "f_min": 0.0
    # })

    # --- Six-Hump Camel Function (2D) ---
    camel6 = (
        (4 - 2.1 * x1**2 + (x1**4) / 3) * x1**2 + x1 * x2 +
        (-4 + 4 * x2**2) * x2**2
    )
    functions.append({
        "name": "Six-Hump Camel",
        "function": camel6,
        "x_start": np.array([0.25, 0.0]),
        "x_min": np.array([0.0898, -0.7126]),  # one of the global minima
        "f_min": -1.0316,
        "x_bounds": [-1.5, 1.5],
        "y_bounds": [-1.5, 1.5]
    })

    return functions


def get_steep_ridge_functions():
    x1, x2 = sym.symbols("x1 x2")
    functions = []

    # --- Easom Function (2D) ---
    easom = -sym.cos(x1) * sym.cos(x2) * sym.exp(-((x1 - sym.pi)**2 + (x2 - sym.pi)**2))
    functions.append({
        "name": "Easom",
        "function": easom,
        "x_start": np.array([2.0, 2.0]),
        "x_min": np.array([np.pi, np.pi]),
        "f_min": -1.0,
        "x_bounds": [1.7, 4.7],
        "y_bounds": [1.7, 4.7]
    })

    # --- Michalewicz Function (n=2, 5, 10) ---
    x_min = [2.20, 1.57]
    f_min = [-1.8013, -4.687658, -9.66015]

    for idx, n in enumerate([2]):
        xs = sym.symbols(f'x1:{n+1}')
        m = 10
        expr = -sum(
            sym.sin(xs[i]) * sym.sin(((i+1) * xs[i]**2) / sym.pi)**(2 * m)
            for i in range(n)
        )
        functions.append({
            "name": f"Michalewicz (n={n})",
            "function": expr,
            "x_start": np.full(n, 2.5),
            "x_min": np.array(x_min),  # global min not known in closed form
            "f_min": f_min[idx],
            "x_bounds": [0.5, 3.5],
            "y_bounds": [0.5, 3.5]
        })

    return functions


def get_other_functions():
    x1, x2, x3, x4 = sym.symbols("x1 x2 x3 x4")
    functions = []

    # --- Perm Function (n=2, 5, 10) ---
    for n in [2, 5, 10]:
        xs = sym.symbols(f'x1:{n + 1}')
        beta = 0.5
        expr = sum([
            (
                sum(
                    (j + 1 + beta) * ((xs[j] / (j + 1)) ** i - 1)
                    for j in range(n)
                )
            ) ** 2
            for i in range(1, n + 1)
        ])
        functions.append({
            "name": f"Perm (n={n})",
            "function": expr,
            "x_start": np.full(n, 0.5),
            "x_min": np.array([i + 1 for i in range(n)]),  # [1, 2, ..., n]
            "f_min": 0.0,
            "x_bounds": [-1.5, 3],
            "y_bounds": [-1.5, 3]
        })

    # --- Powell Function (n=4, 8, 12) ---
    for n in [4, 8, 12]:
        assert n % 4 == 0, "Powell function requires dimension divisible by 4"
        xs = sym.symbols(f'x1:{n+1}')
        expr = 0
        for i in range(0, n, 4):
            xi1, xi2, xi3, xi4 = xs[i], xs[i+1], xs[i+2], xs[i+3]
            expr += (xi1 + 10 * xi2)**2
            expr += 5 * (xi3 - xi4)**2
            expr += (xi2 - 2 * xi3)**4
            expr += 10 * (xi1 - xi4)**4
        functions.append({
            "name": f"Powell (n={n})",
            "function": expr,
            "x_start": np.array([3.0, -1.0, 0.0, 1.0] * (n // 4)),
            "x_min": np.zeros(n),
            "f_min": 0.0
        })

    # --- Goldstein-Price Function (2D) ---
    gp = (
        (1 + (x1 + x2 + 1)**2 * (19 - 14 * x1 + 3 * x1**2 - 14 * x2 + 6 * x1 * x2 + 3 * x2**2)) *
        (30 + (2 * x1 - 3 * x2)**2 * (18 - 32 * x1 + 12 * x1**2 + 48 * x2 - 36 * x1 * x2 + 27 * x2**2))
    )
    functions.append({
        "name": "Goldstein-Price",
        "function": gp,
        "x_start": np.array([0.0, -2.0]),
        "x_min": np.array([0.0, -1.0]),
        "f_min": 3.0,
        "x_bounds": [-2.2, 1],
        "y_bounds": [-2.2, 1]
    })

    # --- Styblinski–Tang Function (n=2, 5, 10) ---
    for n in [2, 5, 10]:
        xs = sym.symbols(f'x1:{n+1}')
        expr = sum(x**4 - 16 * x**2 + 5 * x for x in xs) / 2
        functions.append({
            "name": f"Styblinski-Tang (n={n})",
            "function": expr,
            "x_start": np.array([0.1] + np.full(n-1, 0.1 - 2.7).tolist()),
            "x_min": np.full(n, -2.903534),  # known global min (approx)
            "f_min": n * (-39.16599),  # per dimension, then scaled  -39.16616570377142
            "x_bounds": [-4, 4],
            "y_bounds": [-4, 4]
        })

    return functions


def generate_test_functions():
    return (
        get_plate_shaped_functions() +
        get_valley_shaped_functions() +
        get_steep_ridge_functions() +
        get_other_functions()
    )
