from unittest import TestCase
import numpy as np
from src.unconstrained_min import UnconstrainedMinimization
from src.utils import contour, plot
from tests.examples import Q1, Q2, Q3, rosenbrock, linear, exponential

class TestUnconstrainedMin(TestCase):
    gd_minimization = UnconstrainedMinimization()
    nm_minimization = UnconstrainedMinimization()
    x0 = np.array([1, 1])
    x0_rosenbrock = np.array([-1, 2])
    obj_tol = 10e-8
    param_tol = 10e-12

    def test_Q1(self):
        print("Running test_Q1")

        f = Q1

        gd_x, gd_f_x, gd_success = self.gd_minimization.gradient_descent(f, self.x0, self.obj_tol, self.param_tol)
        nm_x, nm_f_x, nm_success = self.nm_minimization.newton_method(f, self.x0, self.obj_tol, self.param_tol)

        contour(f, self.gd_minimization.x_history, self.nm_minimization.x_history, [-2.5, 2.5], [-2.5, 2.5], 'Q1')
        plot(self.gd_minimization.f_x_history, self.nm_minimization.f_x_history, 'Q1')

        print("Gradient Descent: x = ", gd_x, " f(x) = ", gd_f_x, " Success: ", gd_success)
        print("Newton's Method: x = ", nm_x, " f(x) = ", nm_f_x, " Success: ", nm_success)
    
    def test_Q2(self):
        print("Running test_Q2")

        f = Q2

        gd_x, gd_f_x, gd_success = self.gd_minimization.gradient_descent(f, self.x0, self.obj_tol, self.param_tol)
        nm_x, nm_f_x, nm_success = self.nm_minimization.newton_method(f, self.x0, self.obj_tol, self.param_tol)

        contour(f, self.gd_minimization.x_history, self.nm_minimization.x_history, [-2.5, 2.5], [-2.5, 2.5], 'Q2')
        plot(self.gd_minimization.f_x_history, self.nm_minimization.f_x_history, 'Q2')
        
        print("Gradient Descent: x = ", gd_x, " f(x) = ", gd_f_x, " Success: ", gd_success)
        print("Newton's Method: x = ", nm_x, " f(x) = ", nm_f_x, " Success: ", nm_success)

    def test_Q3(self):
        print("Running test_Q3")

        f = Q3

        gd_x, gd_f_x, gd_success = self.gd_minimization.gradient_descent(f, self.x0, self.obj_tol, self.param_tol)
        nm_x, nm_f_x, nm_success = self.nm_minimization.newton_method(f, self.x0, self.obj_tol, self.param_tol)

        contour(f, self.gd_minimization.x_history, self.nm_minimization.x_history, [-2.5, 2.5], [-2.5, 2.5], 'Q3')
        plot(self.gd_minimization.f_x_history, self.nm_minimization.f_x_history, 'Q3')
        
        print("Gradient Descent: x = ", gd_x, " f(x) = ", gd_f_x, " Success: ", gd_success)
        print("Newton's Method: x = ", nm_x, " f(x) = ", nm_f_x, " Success: ", nm_success)
    
    def test_rosenbrock(self):
        print("Running test_rosenbrock")

        f = rosenbrock

        gd_x, gd_f_x, gd_success = self.gd_minimization.gradient_descent(f, self.x0_rosenbrock, self.obj_tol, self.param_tol, max_iter=10000)
        nm_x, nm_f_x, nm_success = self.nm_minimization.newton_method(f, self.x0_rosenbrock, self.obj_tol, self.param_tol)

        contour(f, self.gd_minimization.x_history, self.nm_minimization.x_history, [-2.5, 2.5], [-2.5, 2.5], 'Rosenbrock')
        plot(self.gd_minimization.f_x_history, self.nm_minimization.f_x_history, 'Rosenbrock')
        
        print("Gradient Descent: x = ", gd_x, " f(x) = ", gd_f_x, " Success: ", gd_success)
        print("Newton's Method: x = ", nm_x, " f(x) = ", nm_f_x, " Success: ", nm_success)
    
    def test_linear(self):
        print("Running test_linear")

        f = linear

        gd_x, gd_f_x, gd_success = self.gd_minimization.gradient_descent(f, self.x0, self.obj_tol, self.param_tol)
        nm_x, nm_f_x, nm_success = self.nm_minimization.newton_method(f, self.x0, self.obj_tol, self.param_tol)

        contour(f, self.gd_minimization.x_history, self.nm_minimization.x_history, [-2.5, 2.5], [-2.5, 2.5], 'Linear Function')
        plot(self.gd_minimization.f_x_history, self.nm_minimization.f_x_history, 'Linear Function')
        
        print("Gradient Descent: x = ", gd_x, " f(x) = ", gd_f_x, " Success: ", gd_success)
        print("Newton's Method: x = ", nm_x, " f(x) = ", nm_f_x, " Success: ", nm_success)

    def test_exponential(self):
        print("Running test_exponential")

        f = exponential

        gd_x, gd_f_x, gd_success = self.gd_minimization.gradient_descent(f, self.x0, self.obj_tol, self.param_tol)
        nm_x, nm_f_x, nm_success = self.nm_minimization.newton_method(f, self.x0, self.obj_tol, self.param_tol)

        contour(f, self.gd_minimization.x_history, self.nm_minimization.x_history, [-2.5, 2.5], [-2.5, 2.5], 'Exponential Function')
        plot(self.gd_minimization.f_x_history, self.nm_minimization.f_x_history, 'Exponential Function')
        
        print("Gradient Descent: x = ", gd_x, " f(x) = ", gd_f_x, " Success: ", gd_success)
        print("Newton's Method: x = ", nm_x, " f(x) = ", nm_f_x, " Success: ", nm_success)
