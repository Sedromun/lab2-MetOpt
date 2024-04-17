import numpy as np

from methods_module.gradient import FunctionNonConvergence, Gradient
from math_module.math_util import *


class Newton:
    def __init__(self, differentiable_function, start_point=None):
        self.differentiable_function = differentiable_function
        self.points = [start_point if (start_point is not None) else (0, 0)]

    @staticmethod
    def rotate_vector(length, a):
        return length * np.cos(a), length * np.sin(a)

    def newton(self):
        while True:
            x, y = self.points[-1]
            if Gradient.vector_length((derivative_x(x, y, self.differentiable_function),
                                       derivative_y(x, y, self.differentiable_function))) < epsilon:
                break
            new_point = (x - derivative_x(x, y, self.differentiable_function)
                         / second_derivative_x(x, y,self.differentiable_function),
                         y - derivative_y(x, y, self.differentiable_function)
                         / second_derivative_y(x, y,self.differentiable_function))
            self.points.append(new_point)
            if len(self.points) > 100 or abs(self.points[-1][0]) > 10000000 or abs(self.points[-1][1]) > 10000000:
                raise FunctionNonConvergence()
        return self.points[len(self.points) - 1]

    def find_minimum(self):
        return self.newton()
