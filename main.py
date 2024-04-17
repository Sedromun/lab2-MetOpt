from typing import Callable

from methods_module.gradient import FunctionNonConvergence
from methods_module.newton import Newton
from visualisation_module.visualisation import *
from methods_module.scipy_methods import *
from methods_module.d1_methods import *
from methods_module.coordinate_descent import *
from random import randint as rand
from math_module.functions import functions
from tabulate import tabulate

points = []


# function from habr
def rosen(x, y):
    return 100.0 * (y - x ** 2.0) ** 2.0 + (1 - x) ** 2.0


def rosen_jac(x, y):
    m1 = -400.0 * x * (y - x ** 2.0) - 2 * (1 - x)
    m2 = 200 * (y - x ** 2.0)
    return [m1, m2]


def rosen_hess(x, y):
    m11 = 1200.0 * x ** 2.0 - 400 * y + 2
    m12 = 0
    m21 = 0
    m22 = 200 * y
    return [[m11, m12],
            [m21, m22]]


def logger(f: Callable[[float, float], float]) -> Callable[[float, float], float]:
    def foo(x: float, y: float) -> float:
        points.append((x, y))
        return f(x, y)

    return foo


def stat_method(
        method: Callable[[Callable[[float, float], float], tuple[float, float]], tuple[float, float]],
        func: FuncWrapper,
):
    attempts, success, semi_success, errors = 100, 0, 0, 0
    for _ in range(attempts):
        start = (rand(-8, 8), rand(-8, 8))
        try:
            x, y = method(func.f, start)
        except Exception:
            errors += 1
        else:
            if abs(func.f(x, y) - func.min) < 0.001:
                success += 1
            elif func.is_infimum(x, y):
                semi_success += 1
                if func.name == "Euclidean distance":
                    print(f"SEMI-SUCCESS: x={x}, y={y}, value={func.f(x, y)}")

    return attempts, success, semi_success, errors


def process_newton(func, start):
    newton = Newton(func.f, start_point=start)
    try:
        x, y = newton.find_minimum()
    except FunctionNonConvergence:
        print('ERROR start point: ', start)
    except Exception as e:
        print('ERROR start point: ', start, " Error:", e)
    else:
        print("NEWTON's METHOD: ", x, y, " Value :=", func.f(x, y))
        draw(newton.points, func.f, x, y, title="Newton's Method")


def process_gradient_descent(func, start):
    grad = Gradient(func, start_point=start)
    try:
        x, y = grad.find_minimum()
    except FunctionNonConvergence:
        print('ERROR start point: ', start)
    else:
        print("GRADIENT DESCENT: ", x, y, " Value :=", func(x, y))
        draw(grad.points, func, x, y, title="Gradient Descent")


def process_d1_search_gradient(func, start):
    grad = Gradient(func, start_point=start, calc_learning_rate=calc_learning_rate)
    try:
        x, y = grad.find_minimum()
    except FunctionNonConvergence:
        print('ERROR start point: ', start)
    else:
        print("GRADIENT DESCENT WITH D1 OPTIMIZATION: ", x, y, " Value :=", func(x, y))
        draw(grad.points, func, x, y, title="Gradient Descent with D1 optimization")


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
    x, y, c_points = coordinate_descent(func, 1, start)
    print("COORDINATE DESCENT: ", x, y, " Value :=", func(x, y))
    draw(c_points, func, x, y, title="Coordinate Descent")


def draw(dots, func, x, y, title: str = ""):
    draw_graphic(dots, func, title=title)
    draw_graphic_2(dots, func, title=title)
    draw_isolines(dots, func, title=title)
    draw_chart(func, (x, y), title=title)


def stat():
    print("NELDER-MEAD")
    headers = ["Function", "Attempts", "Success", "Semi-success", "Incorrect"]
    results = []
    for func in functions:
        a, s, ss, _ = stat_method(nelder_mead, func)
        results.append((func.name, a, s, ss, a - s - ss))
    print(tabulate(results, headers=headers))
    print("\n\n\n")


def run(func, st_point):
    process_newton_cg(func, st_point, rosen_jac, rosen_hess)

    process_BFSG(func, st_point, rosen_jac)

    # process_gradient_descent(func, st_point)
    #
    # process_d1_search_gradient(func, st_point)
    #
    # process_nelder_mead(func, st_point)
    #
    # process_coordinate_descent(func, st_point)


if __name__ == '__main__':
    start_point = (rand(-8, 8), rand(-8, 8))

    # DO NOT DELETE
    # DO NOT DELETE
    # DO NOT DELETE
    # DO NOT DELETE
    # DO NOT DELETE
    # DO NOT DELETE
    # DO NOT DELETE
    # DO NOT DELETE
    # DO NOT DELETE
    # process_nelder_mead(functions[3], start_point)  # Sample of work

    stat()

    # run(functions[0], start_point)  # TODO
