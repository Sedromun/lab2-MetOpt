import math
from typing import Callable

import numpy as np
from pydantic import BaseModel


class FuncWrapper(BaseModel):
    f: Callable[[float, float], float]
    str_value: str
    min: float
    logarithmic: bool
    is_infinum: Callable[[float, float], bool]


def near(x: float, y: float, points: list[tuple[float, float]]) -> bool:
    for point in points:
        if abs(x - point[0]) < 0.0001 and abs(y - point[1]) < 0.0001:
            return True
    return False


functions = [
    FuncWrapper(
        f=lambda x, y: -(x ** 2) + y ** 2 + (x ** 4) / 10,
        str_value="Titties",
        min=-2.5,
        logarithmic=True,
        is_infinium=lambda x, y: near(x, y, [(math.sqrt(5), 0), (0, 0), (-math.sqrt(5), 0)])
    ),
    FuncWrapper(
        f=lambda x, y: -(x ** 2) - (y ** 2) + (x ** 4) / 10 + (y ** 4) / 20 + y + 2 * x,
        str_value="Pig titties",
        min=-15.671267003711808,
        logarithmic=True,
        is_infinium=lambda x, y: near(x, y, [(-2.6273503214528366, -3.387640767635553),
                                             (-2.6273503214528366, 2.8740922205040698)])
    ),
    FuncWrapper(
        f=lambda x, y: -np.exp(-(x ** 2) - (y ** 2)),
        str_value="Bell",
        min=-1,
        logarithmic=True,
        is_infinium=lambda x, y: near(x, y, [(0, 0)])
    ),
    FuncWrapper(
        f=lambda x, y: np.abs(x + y) + 3 * np.abs(y - x),
        str_value="Euclidean distance",
        min=0,
        logarithmic=False,
        is_infinium=lambda x, y: near(x, y, [(0, 0)])
    ),
    FuncWrapper(
        f=lambda x, y: (1 - x) ** 2 + 100 * (y - x ** 2) ** 2,
        str_value="Rosenbrock",
        min=0,
        logarithmic=True,
        is_infinium=lambda x, y: near(x, y, [(1, 1)])
    )
]
