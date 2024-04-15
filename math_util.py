from config import epsilon


def derivative_x(x, y, f):
    return (f(epsilon + x, y) - f(x, y)) / epsilon


def derivative_y(x, y, f):
    return (f(x, y + epsilon) - f(x, y)) / epsilon


def gradient(vector, f):
    x, y = vector
    return derivative_x(x, y, f), derivative_y(x, y, f)


def second_derivative_x(x, y, f):
    return (f(2 * epsilon + x, y) -
            2 * f(epsilon + x, y) +
            f(x, y)) / (epsilon * epsilon)


def second_derivative_y(x, y, f):
    return (f(x, y + 2 * epsilon) -
            2 * f(x, y + epsilon) +
            f(x, y)) / (epsilon * epsilon)

# def jacobian(self, vector):
