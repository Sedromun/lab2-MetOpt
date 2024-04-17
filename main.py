from typing import Callable

from math_module.functions import functions
from methods_module.gradient import FunctionNonConvergence, gradient_descent
from methods_module.newton import newton
from visualisation_module.statistic import sub_stat
from visualisation_module.visualisation import *
from methods_module.scipy_methods import *
from methods_module.d1_methods import *
from methods_module.coordinate_descent import *
from random import randint as rand
from methods_module.my_bfgs import my_bfgs

points = []


def logger(f: Callable[[float, float], float]) -> Callable[[float, float], float]:
    def foo(x: float, y: float) -> float:
        points.append((x, y))
        return f(x, y)

    return foo


def process_newton(func, start):
    try:
        x, y = newton(func.f, start_point=start, calc_learning_rate=(lambda a, b, c, d: 1))
    except FunctionNonConvergence:
        print('ERROR start point: ', start)
    except Exception as e:
        print('ERROR start point: ', start, " Error:", e)
    else:
        print("NEWTON's METHOD: ", x, y, " Value :=", func.f(x, y))
        # draw(newton_points, func, x, y, title="Newton's Method")


def process_d1_search_newton(func, start):
    try:
        x, y = newton(func.f, start_point=start, calc_learning_rate=calc_learning_rate)
    except FunctionNonConvergence:
        print('ERROR start point: ', start)
    except Exception as e:
        print('ERROR start point: ', start, " Error:", e)
        raise e
    else:
        print("NEWTON's METHOD WITH D1 OPTIMIZATION: ", x, y, " Value :=", func.f(x, y))
        # draw(newton_points, func, x, y, title="Newton's Method with D1 optimization")


def process_gradient_descent(func, start):
    try:
        x, y = gradient_descent(func.f, start_point=start)
    except FunctionNonConvergence:
        print('ERROR start point: ', start)
    else:
        print("GRADIENT DESCENT: ", x, y, " Value :=", func.f(x, y))
        # draw(gradient_points, func, x, y, title="Gradient Descent")


def process_d1_search_gradient(func, start):
    try:
        gradient_points = gradient_descent(func, start_point=start, calc_learning_rate=calc_learning_rate)
        x, y = gradient_points[-1]
    except FunctionNonConvergence:
        print('ERROR start point: ', start)
    else:
        print("GRADIENT DESCENT WITH D1 OPTIMIZATION: ", x, y, " Value :=", func(x, y))
        draw(gradient_points, func, x, y, title="Gradient Descent with D1 optimization")


def process_nelder_mead(func, start):
    x, y = nelder_mead(logger(func.f), start)
    print("NELDER-MEAD: ", x, y, " Value :=", func.f(x, y))
    draw(points, func, x, y, title="Nelder-Mead")


def process_newton_cg(func, start, jac, hess):
    x, y = newton_cg(logger(func.f), start, jac, hess)
    print("NEWTON-CG: ", x, y, " Value :=", func(x, y))
    draw(points, func, x, y, title="Newton-CG")


def process_BFSG(func, start, jac):
    x, y = BFSG(logger(func.f), start, jac)
    print("BFSG: ", x, y, " Value :=", func(x, y))
    draw(points, func, x, y, title="BFSG")


def process_coordinate_descent(func, start):
    x, y, c_points = coordinate_descent(func, start)
    print("COORDINATE DESCENT: ", x, y, " Value :=", func(x, y))
    draw(c_points, func, x, y, title="Coordinate Descent")


def draw(dots, func, x, y, title: str = ""):
    draw_graphic(dots, func, title=title)
    draw_graphic_2(dots, func, title=title)
    draw_isolines(dots, func, title=title)
    draw_chart(func, (x, y), title=title)


def stat():
    sub_stat(gradient_descent, "GRADIENT DESCENT")
    sub_stat(lambda f, p: gradient_descent(f, p, calc_learning_rate), "GRADIENT DESCENT D1")
    sub_stat(coordinate_descent, "COORDINATE DESCENT")
    sub_stat(nelder_mead, "NELDER MEAD")

    sub_stat(newton, "NEWTON")
    sub_stat(lambda f, p: newton(f, p, calc_learning_rate), "NEWTON WITH D1 OPTIMIZATION")
    sub_stat(newton_cg, "NEWTON-CG")
    sub_stat(my_bfgs, "JEKA's BFSG")
    sub_stat(BFSG, "BFSG")


if __name__ == '__main__':
    start_point = (rand(-8, 8), rand(-8, 8))

    process_d1_search_newton(functions[0], start_point)  # Sample of work

    # stat()
