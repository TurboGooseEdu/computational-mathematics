import numpy as np
from ode.runge_kutta_method import RungeKuttaMethod

A = 3.
B = -3.
C = 3.
c2 = 0.35
x0 = 0.
Y0 = np.array([1., 1., A, 1.])
x_start = 0
x_fin = 5


def generate_RKM2(c):
    return RungeKuttaMethod([0, c], [[0, 0], [c, 0]], [1 - 1 / (2 * c), 1 / (2 * c)], 2)


considered_method = generate_RKM2(c2)
opponent_method = generate_RKM2(1/2)  # midpoint


def exact_solution(x):
    return np.array([
        np.exp(np.sin(x ** 2)),
        np.exp(B * np.sin(x ** 2)),
        C * np.sin(x ** 2) + A,
        np.cos(x ** 2)
    ])


def F(x, y):
    return np.array([
        2 * x * y[1]**(1/B) * y[3],
        2 * B * x * np.exp(B / C * (y[2] - A)) * y[3],
        2 * C * x * y[3],
        -2 * x * np.log(y[0])
    ])


def solve_constant_step(start, stop, h, method):
    Y = Y0
    x_values = []
    Y_values = [Y]
    for x in np.arange(start, stop, h):
        Y = method.calculate(F, h, x, Y)
        x_values.append(x)
        Y_values.append(Y)
    if x_values[-1] < stop:
        x_values.append(stop)
    else:
        del Y_values[-1]
    return x_values, Y_values


def eval_error_by_Runge(Yn, Y2n, method):
    p = method.precision
    R2n = (Y2n - Yn) / (2**p - 1)
    return R2n


def solve_auto_step(start, stop, method):
    h = 0.02
    tol = 1.e-6
    factor = 0.9
    facmin = 0.3
    facmax = 3
    p = method.precision
    Y_cur = Y0
    x_cur = start
    x_values = [x_cur]
    x_discarded = []
    Y_values = [Y_cur]
    while x_cur < stop:
        Yn = method.calculate(F, h, x_cur, Y_cur)
        Y_mid = method.calculate(F, h/2, x_cur, Y_cur)
        Y2n = method.calculate(F, h/2, x_cur + h/2, Y_mid)
        R2n = eval_error_by_Runge(Yn, Y2n, method)
        R2n_norm = np.linalg.norm(R2n)
        if R2n_norm < tol:
            Y_cur = Y2n + R2n
            x_cur += h
            x_values.append(x_cur)
            Y_values.append(Y_cur)
        elif len(x_discarded) == 0 or x_cur != x_discarded[-1]:
            x_discarded.append(x_cur)
        h = min(stop - x_cur, h * max(facmin, min(facmax, factor * (tol / R2n_norm)**(1/p))))
    return x_values, Y_values, x_discarded
