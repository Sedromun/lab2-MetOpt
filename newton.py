import math

import numpy as np

from config import epsilon
from gradient import FunctionNonConvergence, Gradient


class Newton:
    def __init__(self, differentiable_function, start_point=None):
        self.differentiable_function = differentiable_function
        self.points = [start_point if (start_point is not None) else (0, 0)]

    @staticmethod
    def rotate_vector(length, a):
        return length * np.cos(a), length * np.sin(a)

    def derivative_x(self, x, y):
        return (self.differentiable_function(epsilon + x, y) -
                self.differentiable_function(x, y)) / epsilon

    def derivative_y(self, x, y):
        return (self.differentiable_function(x, y + epsilon) -
                self.differentiable_function(x, y)) / epsilon

    def second_derivative_x(self, x, y):
        return (self.differentiable_function(2 * epsilon + x, y) -
                2 * self.differentiable_function(epsilon + x, y) +
                self.differentiable_function(x, y)) / (epsilon * epsilon)

    def second_derivative_y(self, x, y):
        return (self.differentiable_function(x, y + 2 * epsilon) -
                2 * self.differentiable_function(x, y + epsilon) +
                self.differentiable_function(x, y)) / (epsilon * epsilon)

    def newton(self):

        while True:
            x, y = self.points[-1]
            if Gradient.vector_length((self.derivative_x(x, y), self.derivative_y(x, y))) < epsilon:
                break
            new_point = (x - self.derivative_x(x, y) / self.second_derivative_x(x, y),
                         y - self.derivative_y(x, y) / self.second_derivative_y(x, y))
            self.points.append(new_point)
            if len(self.points) > 100 or abs(self.points[-1][0]) > 10000000 or abs(self.points[-1][1]) > 10000000:
                raise FunctionNonConvergence()
        return self.points[len(self.points) - 1]

    def find_minimum(self):
        return self.newton()
