import scipy


def nelder_mead(f, start_point):
    def n_m_f(x):
        return f(x[0], x[1])

    res = scipy.optimize.minimize(n_m_f, start_point, method='Nelder-Mead')
    return res.x[0], res.x[1]


# jac - функция, вычисляющая матрицу первых производных
# hess - функция, вычисляющая матрицу вторых производных
def newton_cg(f, start_point, jac):
    def n_m_f(x):
        return f(x[0], x[1])
    def j (x):
        return jac(x[0], x[1])

    res = scipy.optimize.minimize(n_m_f, start_point, method='Newton-CG', jac=j)
    return res.x[0], res.x[1]
