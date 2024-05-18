from typing import Callable, Tuple
import numpy as np
from numpy.typing import ArrayLike

class UnconstrainedMinimization:
    def __init__(self) -> None:
        """
        Initialize the unconstrained minimization object.
        """
        self.x_history = []
        self.f_x_history = []

    def newton_method(
        self,
        f: Callable[[ArrayLike], bool],
        x0: ArrayLike,
        obj_tol: float,
        param_tol: float,
        max_iter: int = 100
    ) -> Tuple[ArrayLike, float, bool]:
        """
        Unconstrained minimization using Newton's method.

        Args:
            f: The function to minimize.
            x0: Starting point for the minimization.
            obj_tol: Numeric tolerance for successful termination in terms of small enough change in objective function values.
            param_tol: Numeric tolerance for successful termination in terms of small enough distance between two consecutive iteration locations.
            max_iter: Maximum allowed number of iterations.
            
        Returns:
            Tuple[ArrayLike, float, bool]: A tuple containing:
                - ArrayLike: Final location.
                - float: Final objective value.
                - bool: A boolean flag indicating successful termination.
        """
        x = x0

        # Get the initial objective function value, gradient, and Hessian matrix
        f_x, grad, hess = f(x, hess=True)

        # Store the initial location and objective function value
        self.x_history.append(x)
        self.f_x_history.append(f_x)

        # Iterate until the maximum number of iterations is reached
        for _ in range(max_iter):
            # Get the search direction and step size
            f_x, grad, hess = f(x, hess=True)
            try:
                direction = np.linalg.solve(hess, -grad)
            except np.linalg.LinAlgError:
                print("Hessian matrix is singular. Newton's method cannot proceed.")
                return x, f_x, False
            step = get_step(x, f, f_x, direction)

            # Update the location based on the search direction and step size
            x_new = x + step * direction

            # Get the objective function value and Hessian matrix at the new location
            f_x_new, _, hess = f(x_new, hess=True)

            # Check for convergence
            lambda_ = np.sqrt(direction.T @ hess @ direction)
            if np.linalg.norm(x_new - x) < param_tol or (lambda_**2 * 0.5) < obj_tol:
                return x_new, f_x_new, True
            
            # Update the location and objective function value
            x = x_new
            f_x = f_x_new

            # Store the location and objective function value
            self.x_history.append(x)
            self.f_x_history.append(f_x)

        # Return the final location and objective function value, and a flag indicating unsuccessful termination
        return x, f_x, False


    def gradient_descent(
        self,
        f: Callable[[ArrayLike], bool],
        x0: ArrayLike,
        obj_tol: float,
        param_tol: float,
        max_iter: int = 100
    ) -> Tuple[ArrayLike, float, bool]:
        """
        Unconstrained minimization using gradient descent.

        Args:
            f: The function to minimize.
            x0: Starting point for the minimization.
            obj_tol: Numeric tolerance for successful termination in terms of small enough change in objective function values.
            param_tol: Numeric tolerance for successful termination in terms of small enough distance between two consecutive iteration locations.
            max_iter: Maximum allowed number of iterations.
            
        Returns:
            Tuple[ArrayLike, float, bool]: A tuple containing:
                - ArrayLike: Final location.
                - float: Final objective value.
                - bool: A boolean flag indicating successful termination.
        """
        x = x0

        # Get the initial objective function value and gradient
        f_x, grad, _ = f(x)

        # Store the initial location and objective function value
        self.x_history.append(x)
        self.f_x_history.append(f_x)

        # Iterate until the maximum number of iterations is reached
        for _ in range(max_iter):
            # Get the search direction and step size
            direction = -grad
            step = get_step(x, f, f_x, direction)

            # Update the location based on the search direction and step size
            x_new = x + step * direction

            # Get the objective function value at the new location
            f_x_new, _, _ = f(x_new)

            # Check for convergence
            if np.linalg.norm(x_new - x) < param_tol or abs(f_x_new - f_x) < obj_tol:
                return x_new, f_x_new, True
            
            # Update the location and objective function value
            x = x_new
            f_x = f_x_new

            # Store the location and objective function value
            self.x_history.append(x)
            self.f_x_history.append(f_x)

        # Return the final location and objective function value, and a flag indicating unsuccessful termination
        return x, f_x, False


def get_step(x: ArrayLike, f: Callable[[ArrayLike], bool], f_x: float, direction: ArrayLike, max_iter: int = 1000) -> int:
    """
    Method to get the step size for the line search.

    Args:
        x: The current location.
        f: The function to minimize.
        f_x: The value of the objective function at x.
        direction: The search direction.
        max_iter: The maximum number of iterations to perform.
        
    Returns:
        int: The step size.
    """
    wolfe_const = 0.01
    backtracking_const = 0.5

    step = 1
    i = 1
    while f(x + step * direction)[0] > f_x + wolfe_const * step * np.dot(-direction, direction) and i < max_iter:
        step *= backtracking_const
        i += 1

    return step