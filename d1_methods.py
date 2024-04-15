# вычисляет минимум на отрезке [a, b] с точностью epsilon для унимодальных функций
from config import epsilon, learning_rate
from gradient import Gradient


def calc_learning_rate(f, x: tuple, gradient: tuple):
    def g(t: float):
        vector = Gradient.vectors_subtract(x, Gradient.mult_vector(t, gradient))
        return f(vector[0], vector[1])

    return dichotomy_method(g, 0, learning_rate)


def dichotomy_method(f, a, b):
    if f(a) * f(b) > 0:
        return b

    while (b - a) / 2 > epsilon:
        c = (a + b) / 2
        if f(c) == 0:
            return c
        elif f(a) * f(c) < 0:
            b = c
        else:
            a = c
    return (a + b) / 2
