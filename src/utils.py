from typing import Callable, Tuple
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike

def contour(
    f: Callable[[ArrayLike], bool],
    x_history_gd: ArrayLike,
    x_history_nm: ArrayLike,
    x_limit: Tuple[float, float],
    y_limit: Tuple[float, float],
    title: str
) -> None:
    """
    Plot the contour of the objective function and the trajectory of the optimization algorithms.
    
    Args:
        f: The objective function to plot.
        x_history_gd: The history of the optimization algorithm using gradient descent.
        x_history_nm: The history of the optimization algorithm using Newton's method.
        x_limit: The limits of the x-axis.
        y_limit: The limits of the y-axis.
        title: The title of the plot.
    """
    x_points_gd = np.array(x_history_gd)[:, 0]
    y_points_gd = np.array(x_history_gd)[:, 1]

    x_points_nm = np.array(x_history_nm)[:, 0]
    y_points_nm = np.array(x_history_nm)[:, 1]

    X = np.linspace(x_limit[0], x_limit[1], 100)
    Y = np.linspace(y_limit[0], y_limit[1], 100)
    X, Y = np.meshgrid(X, Y)
    Z = np.zeros(X.shape)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = f([X[i, j], Y[i, j]])[0]

    plt.figure()
    plt.contour(X, Y, Z, levels=20, title=title)

    plt.plot(x_points_gd, y_points_gd, 'xr-', label='Gradient Descent')
    plt.plot(x_points_nm, y_points_nm, 'xb-', label="Newton's Method")

    plt.legend()
    plt.title(label=title)
    plt.show()

def plot(
    f_x_history_gd: ArrayLike,
    f_x_history_nm: ArrayLike,
    title: str
) -> None:
    """
    Plot the objective function values over iterations.
    
    Args:
        f_x_history_gd: The history of the objective function values using gradient descent.
        f_x_history_nm: The history of the objective function values using Newton's method.
        title: The title of the plot.
    """
    iter_gd = []
    for i in range(len(f_x_history_gd)):
        iter_gd.append(i)

    iter_nm = []
    for i in range(len(f_x_history_nm)):
        iter_nm.append(i)


    plt.plot(iter_gd, f_x_history_gd, 'r-', label='Gradient Descent')
    plt.plot(iter_nm, f_x_history_nm, 'b-', label="Newton's Method")

    plt.legend()
    plt.title(label=title)
    plt.show()