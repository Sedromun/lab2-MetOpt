import math

import numpy as np

from config import learning_rate, epsilon, centre, arr_shape


class FunctionNonConvergence(Exception):
    def __init__(self):
        super(FunctionNonConvergence, self)


class Gradient:
    def __init__(self, differentiable_function, start_point=None, calc_learning_rate=(lambda x, y, z: learning_rate)):
        self.differentiable_function = differentiable_function
        self.points = [start_point if (start_point is not None) else (0, 0)]
        self.calc_learning_rate = calc_learning_rate

    @staticmethod
    def rotate_vector(length, a):
        return length * np.cos(a), length * np.sin(a)

    def derivative_x(self, x, y):
        return (self.differentiable_function(epsilon + x, y) -
                self.differentiable_function(x, y)) / epsilon

    def derivative_y(self, x, y):
        return (self.differentiable_function(x, y + epsilon) -
                self.differentiable_function(x, y)) / epsilon

    @staticmethod
    def vector_length(vector):
        a, b = vector
        return math.sqrt(a ** 2 + b ** 2)

    @staticmethod
    def vectors_subtract(vector1, vector2):
        x1, y1 = vector1
        x2, y2 = vector2
        return (x1 - x2), (y1 - y2)

    @staticmethod
    def mult_vector(num, vector):
        x, y = vector
        return x * num, y * num

    def gradient(self, vector):
        x, y = vector
        return self.derivative_x(x, y), self.derivative_y(x, y)

    def gradient_descent(self):
        while (len(self.points) < 2 or self.vector_length(
                self.vectors_subtract(
                    self.points[len(self.points) - 1],
                    self.points[len(self.points) - 2]
                )
        ) > epsilon):
            grad = self.gradient(self.points[len(self.points) - 1])
            self.points.append(
                self.vectors_subtract(
                    self.points[len(self.points) - 1],
                    self.mult_vector(self.calc_learning_rate(
                        self.differentiable_function,
                        self.points[len(self.points) - 1],
                        grad), grad)
                ))
            if len(self.points) > 100000 or abs(self.points[-1][0]) > 10000000 or abs(self.points[-1][1]) > 10000000:
                raise FunctionNonConvergence()
        return self.points[len(self.points) - 1]

    def find_minimum(self):
        return self.gradient_descent()
