from config import epsilon


def derivative_x(x, y, f):
    return (f(epsilon + x, y) - f(x, y)) / epsilon


def derivative_y(x, y, f):
    return (f(x, y + epsilon) - f(x, y)) / epsilon


def gradient(vector, f):
    x, y = vector
    return derivative_x(x, y, f), derivative_y(x, y, f)

# def jacobian(self, vector):
