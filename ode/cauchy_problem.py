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
tol = 1.e-5


def generate_RKM2(c):
    return RungeKuttaMethod([0, c], [[0, 0], [c, 0]], [1 - 1 / (2 * c), 1 / (2 * c)], 2)


considered_method = generate_RKM2(c2)
opponent_method = generate_RKM2(1/2)  # midpoint


def y1(x):
    return np.exp(np.sin(x**2))


def y2(x):
    return np.exp(B * np.sin(x**2))


def y3(x):
    return C * np.sin(x**2) + A


def y4(x):
    return np.cos(x**2)


def F(x, y):
    return np.array([
        2 * x * y[1]**(1/B) * y[3],
        2 * B * x * np.exp(B / C * (y[2] - A)) * y[3],
        2 * C * x * y[3],
        -2 * x * np.log(y[0])
    ])


def solve_constant_step(method, h):
    N = int((x_fin - x_start) / h)
    x = x0
    Y = Y0
    x_values = [x]
    Y_values = [Y]
    for i in range(N):
        Y = method.calculate(F, h, x, Y)
        print("{}) {}".format(i + 1, Y))
        x += h
        x_values.append(x)
        Y_values.append(Y)
    if x < x_fin:
        x_values.append(x_fin)
        Y_values.append(method.calculate(F, x_fin - x, x, Y))

    return x_values, Y_values
