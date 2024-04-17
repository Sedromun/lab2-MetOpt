import scipy


def nelder_mead(f, start_point):
    def n_m_f(x):
        return f(x[0], x[1])

    res = scipy.optimize.minimize(n_m_f, start_point, method='Nelder-Mead')
    return res.x[0], res.x[1]


# jac - функция, вычисляющая матрицу первых производных
# hess - функция, вычисляющая матрицу вторых производных
def newton_cg(f, start_point, jac, hess):
    def scipy_f(x):
        return f(x[0], x[1])

    def scipy_j(x):
        return jac(x[0], x[1])

    def scipy_h(x):
        return hess(x[0], x[1])

    res = scipy.optimize.minimize(scipy_f, start_point, method='Newton-CG', jac=scipy_j, hess=scipy_h)
    return res.x[0], res.x[1]


# квазиньютоновский 1
def BFSG(f, start_point, jac):
    def scipy_f(x):
        return f(x[0], x[1])

    def scipy_j(x):
        return jac(x[0], x[1])

    res = scipy.optimize.minimize(scipy_f, start_point, method='BFGS', jac=scipy_j)
    return res.x[0], res.x[1]
