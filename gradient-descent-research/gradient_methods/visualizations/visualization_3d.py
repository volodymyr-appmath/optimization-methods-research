import numpy as np
import sympy as sym
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


def plot_3d(func, history, xlim_=(-2, 2), ylim_=(-2, 2)):
    # Converting sympy function to lambda expression
    variables = list(sym.ordered(func.free_symbols))
    function = sym.lambdify(variables, func)

    # Generate a grid of points
    x_vals = np.linspace(*xlim_, 1000)
    y_vals = np.linspace(*ylim_, 1000)
    X, Y = np.meshgrid(x_vals, y_vals)
    F = function(X, Y)

    # Create a custom colormap
    hot_colormap = plt.get_cmap('hot')
    hot_colors = hot_colormap(np.linspace(0, 1, 256))

    # Set values below a certain threshold to blue
    threshold = 1e-10
    below_threshold_color = np.array([0, 0, 1, 1])  # RGBA for blue

    # Find the index where the colormap should start changing
    threshold_index = np.searchsorted(np.linspace(0, np.max(F), 256), threshold)

    # Modify the colormap
    new_colors = hot_colors.copy()
    new_colors[:threshold_index] = below_threshold_color

    # Create a new colormap with the modified colors
    custom_colormap = ListedColormap(new_colors)

    # 3D plot of a function
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    surface = ax.plot_surface(X, Y, F, cmap=custom_colormap, edgecolor='none', zorder=4)

    # Add a color bar which maps values to colors
    fig.colorbar(surface)

    # Add labels and title
    ax.set_title('Rosenbrock Function')
    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$f \ (x_1, x_2)$', rotation=90)

    points = np.array(history)#.squeeze(axis=2)

    # Calculate the corresponding f(x1, x2) values
    f_values = function(points[:, 0], points[:, 1])

    # Connect the points with straight lines
    ax.plot(points[:, 0], points[:, 1], f_values, color='#bebebe', linestyle='--',
            linewidth=1.2, markersize=3, marker='o', label='CGD path', zorder=5)

    # Add a legend
    ax.legend()

    plt.show()