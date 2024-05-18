from typing import Tuple
import numpy as np
from numpy.typing import ArrayLike

def Q1(x: ArrayLike, hess: bool = False) -> Tuple[float, ArrayLike, ArrayLike]:
    """
    Method to get an objective function to minimize.
    For this one, we're trying the minimize f(x) = x^T Q x, where Q is as follows:
    Q = [[1, 0], [0, 1]] (the contour lines are circles)
    
    Args:
        x: The point at which to evaluate the function.
        hess: A flag indicating whether to return the Hessian matrix.

    Returns:
        Tuple[float, ArrayLike, ArrayLike]: A tuple containing:
            - float: The value of the objective function at x.
            - ArrayLike: The gradient of the objective function at x.
            - ArrayLike: The Hessian matrix of the objective function at x, if hessian is True.
    """
    Q = np.array([[1, 0], [0, 1]])

    f = x @ Q @ x
    grad = 2 * Q @ x

    if hess:
        H = 2 * Q
    else:
        H = None
    
    return f, grad, H

def Q2(x: ArrayLike, hess: bool = False) -> Tuple[float, ArrayLike, ArrayLike]:
    """
    Method to get an objective function to minimize.
    For this one, we're trying the minimize f(x) = x^T Q x, where Q is as follows:
    Q = [[1, 0], [0, 100]] (the contour lines are circles)

    Args:
        x: The point at which to evaluate the function.
        hess: A flag indicating whether to return the Hessian matrix.

    Returns:
        Tuple[float, ArrayLike, ArrayLike]: A tuple containing:
            - float: The value of the objective function at x.
            - ArrayLike: The gradient of the objective function at x.
            - ArrayLike: The Hessian matrix of the objective function at x, if hessian is True.
    """
    Q = np.array([[1, 0], [0, 100]])
    
    f = x @ Q @ x
    grad = 2 * Q @ x

    if hess:
        H = 2 * Q
    else:
        H = None
    
    return f, grad, H

def Q3(x: ArrayLike, hess: bool = False) -> Tuple[float, ArrayLike, ArrayLike]:
    """
    Method to get an objective function to minimize.
    For this one, we're trying the minimize f(x) = x^T Q x, where Q is as follows:
    Q = [[sqrt(3)/2, -0.5], [0.5, sqrt(3)/2]]^T [[100, 0], [0, 1]] [[sqrt(3)/2, -0.5], [0.5, sqrt(3)/2]] (the contour lines are rotated ellipses)

    Args:
        x: The point at which to evaluate the function.
        hess: A flag indicating whether to return the Hessian matrix.

    Returns:
        Tuple[float, ArrayLike, ArrayLike]: A tuple containing:
            - float: The value of the objective function at x.
            - ArrayLike: The gradient of the objective function at x.
            - ArrayLike: The Hessian matrix of the objective function at x, if hessian is True.
    """
    Q = np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]]) @ np.array([[100, 0], [0, 1]]) @ np.array([[np.sqrt(3)/2, -0.5], [0.5, np.sqrt(3)/2]])

    f = x @ Q @ x
    grad = 2 * Q @ x

    if hess:
        H = 2 * Q
    else:
        H = None
    
    return f, grad, H

def rosenbrock(x: ArrayLike, hess: bool = False) -> Tuple[float, ArrayLike, ArrayLike]:
    """
    Method to get an objective function to minimize.
    For this one, we're trying the minimize f(x) = 100*(x[1] - x[0]^2)^2 + (1 - x[0])^2 (the contour lines are banana-shaped ellipses)

    Args:
        x: The point at which to evaluate the function.
        hess: A flag indicating whether to return the Hessian matrix.

    Returns:
        Tuple[float, ArrayLike, ArrayLike]: A tuple containing:
            - float: The value of the objective function at x.
            - ArrayLike: The gradient of the objective function at x.
            - ArrayLike: The Hessian matrix of the objective function at x, if hessian is True.
    """
    f = 100 * (x[1] - x[0]**2)**2 + (1 - x[0])**2
    grad = np.array([2 * (200 * x[0]**3 - 200 * x[0] * x[1] + x[0] - 1), 200 * (x[1] - x[0]**2)])

    if hess:
        H = np.array([[1200 * x[0]**2 - 400 * x[1] + 2, -400 * x[0]], [-400 * x[0], 200]])
    else:
        H = None
    
    return f, grad.T, H

def linear(x: ArrayLike, hess: bool = False) -> Tuple[float, ArrayLike, ArrayLike]:
    """
    Method to get an objective function to minimize.
    For this one, we're trying the minimize f(x) = a^T x for some nonzero vector a (the contour lines are straight lines)

    Args:
        x: The point at which to evaluate the function.
        hess: A flag indicating whether to return the Hessian matrix.

    Returns:
        Tuple[float, ArrayLike, ArrayLike]: A tuple containing:
            - float: The value of the objective function at x.
            - ArrayLike: The gradient of the objective function at x.
            - ArrayLike: The Hessian matrix of the objective function at x, if hessian is True.
    """
    a = np.array([4, 3])

    f = a @ x
    grad = a

    if hess:
        H = np.zeros((2, 2))
    else:
        H = None
    
    return f, grad, H

def exponential(x: ArrayLike, hess: bool = False) -> Tuple[float, ArrayLike, ArrayLike]:
    """
    Method to get an objective function to minimize.
    For this one, we're trying the minimize f(x) = exp(x[0] + 3*x[1] - 0.1) + exp(x[0] - 3*x[1] - 0.1) + exp(-x[0] - 0.1) (the contour lines are smoothed corner triangles)

    Args:
        x: The point at which to evaluate the function.
        hess: A flag indicating whether to return the Hessian matrix.

    Returns:
        Tuple[float, ArrayLike, ArrayLike]: A tuple containing:
            - float: The value of the objective function at x.
            - ArrayLike: The gradient of the objective function at x.
            - ArrayLike: The Hessian matrix of the objective function at x, if hessian is True.
    """
    f1 = np.exp(x[0] + 3*x[1] - 0.1)
    f2 = np.exp(x[0] - 3*x[1] - 0.1)
    f3 = np.exp(-x[0] - 0.1)

    f = f1 + f2 + f3
    grad = np.array([f1 + f2 - f3, 3*f1 - 3*f2])

    if hess:
        H = np.array([[f, 3*(f1 - f2)], [3*(f1 - f2), 9*(f1 + f2)]])
    else:
        H = None
    
    return f, grad.T, H
    