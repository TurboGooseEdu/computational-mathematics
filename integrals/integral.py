from math import sin, cos, exp, inf, log, ceil
from slae.matrix import Matrix
from integrals.lu_decomposer import LUDecomposer
import sympy as sp
import pandas as pd
from IPython.display import display

a = 2.1
b = 3.3
alpha = 2/5
J_exact = 4.461512705331194112840828080521604042844


def f(x):
    return 4.5 * cos(7 * x) * exp(-2/3 * x) + 1.4 * sin(1.5 * x) * exp(-x / 3) + 3


def M0(x1, x2):
    g = lambda x: (x-a)**(1-alpha) / (1-alpha)
    return g(x2) - g(x1)


def M1(From, to):
    g = lambda x: (x-a)**(1-alpha) * (a-alpha*x+x) / ((alpha-2)*(alpha-1))
    return g(to) - g(From)


def M2(From, to):
    g = lambda x: -(x-a)**(1-alpha) * (2*a**2-2*a*(alpha-1)*x+(alpha**2-3*alpha+2)*x**2) / ((alpha-3)*(alpha-2)*(alpha-1))
    return g(to) - g(From)


def M3(From, to):
    g = lambda x: ((x-a)**(1-alpha) * (6*a**3-6*a**2*(alpha-1)*x+3*a*(alpha**2-3*alpha+2)*x**2-(alpha**3-6*alpha**2+11*alpha-6)*x**3)) / ((alpha-4)*(alpha-3)*(alpha-2)*(alpha-1))
    return g(to) - g(From)


def M4(From, to):
    g = lambda x: (x-a)**(1-alpha) * (a**4/(1-alpha) + 4*a**3*(a-x)/(alpha-2) - 6*a**2*(a-x)**2/(alpha-3) + 4*a*(a-x)**3/(alpha-4) - (a-x)**4/(alpha-5))
    return g(to) - g(From)


def M5(From, to):
    g = lambda x: (x-a)**(1-alpha) * (a**5/(1-alpha) + 5*a**4*(a-x)/(alpha-2) - 10*a**3*(a-x)**2/(alpha-3) + 10*a**2*(a-x)**3/(alpha-4) - 5*a*(a-x)**4/(alpha-5) + (a-x)**5/(alpha-6))
    return g(to) - g(From)


def calculate_integral_newton_kotes(eps=1.e-6):
    integral_with_m_range(2.5, 4.5, QF_newton_kotes, eps)


def calculate_integral_gauss(eps=1.e-6):
    integral_with_m_range(5, 7, QF_gauss, eps)


def integral_with_m_range(m1, m2, QF, eps=1.e-6):
    L = 2
    k = 1  # dividing [a, b] for k equal parts
    S = []
    m = inf
    error = inf
    k_opt_calculated = False
    data = {"k": [], "h": [], "m": [], "S": [], "R": [], "|J_exact - S|": []}
    while error > eps:
        S.append(composite_QF(QF, k))
        if len(S) >= 3:
            value = (S[-2] - S[-3]) / (S[-1] - S[-2])
            if value > 0:
                m = log(value, L)
                if m1 < m < m2:
                    error = (S[-1] - S[-2]) / (L**m - 1)
                    if not k_opt_calculated:
                        h_opt = (b - a) / k * (eps / abs(error)) ** (1 / m)
                        k_opt = ceil((b - a) / (h_opt * 0.95))
                        k_opt_calculated = True
                        if k_opt > k:
                            k = k_opt
                        print("k_opt = {}  (h_opt = {})".format(k_opt, h_opt))
        data["k"].append(k)
        data["h"].append((b - a) / k)
        data["m"].append(m)
        data["S"].append(S[-1])
        data["R"].append(error)
        data["|J_exact - S|"].append(abs(J_exact - S[-1]))
        k *= L

    result = S[-1] + error
    display(pd.DataFrame(data))
    print("\nJ       =", result)
    print("J_exact =", J_exact)
    print("|J - J_exact| =", abs(result - J_exact))
    return result


def composite_QF(QF, k):
    h = (b - a) / k
    result = 0
    for i in range(k):
        left = a + i * h
        right = a + (i + 1) * h
        S, R = QF(left, right)
        result += S
    return result


def QF_newton_kotes(From, to):
    return IQF(From, From, (From + to) / 2, to, to)


def QF_gauss(From, to):
    m0 = M0(From, to)
    m1 = M1(From, to)
    m2 = M2(From, to)
    m3 = M3(From, to)
    m4 = M4(From, to)
    m5 = M5(From, to)
    A = Matrix([[m0, m1, m2], [m1, m2, m3], [m2, m3, m4]])
    B = -Matrix([[m3], [m4], [m5]])
    coefs = LUDecomposer(A).solve(B)
    roots = solve_cubic_equation(1, coefs[2][0], coefs[1][0], coefs[0][0])
    return IQF(From, *roots, to)


def solve_cubic_equation(a3, a2, a1, a0):
    x = sp.Symbol("x")
    eq = a3*x**3 + a2*x**2 + a1*x + a0
    return sp.solveset(eq).args


def IQF(From, z1, z2, z3, to):
    m0 = M0(From, to)
    m1 = M1(From, to)
    m2 = M2(From, to)
    A = Matrix([[1, 1, 1], [z1, z2, z3], [z1**2, z2**2, z3**2]])
    B = Matrix([[m0], [m1], [m2]])
    coefs = LUDecomposer(A).solve(B)
    A1 = coefs[0][0]
    A2 = coefs[1][0]
    A3 = coefs[2][0]
    return A1 * f(z1) + A2 * f(z2) + A3 * f(z3), IQF_evaluation(From, z1, z2, z3, to)


def IQF_evaluation(From, z1, z2, z3, to):
    g = lambda x: (x - a) ** (1 - alpha) * (
                ((a - x) * (3 * a ** 2 - 2 * a * (z1 + z2 + z3) + z1 * (z2 + z3) + z2 * z3)) / (alpha - 2) - (
                    (a - x) ** 2 * (3 * a - z1 - z2 - z3)) / (alpha - 3) + (a - x) ** 3 / (alpha - 4) - (
                    (a - z1) * (a - z2) * (a - z3)) / (alpha - 1))
    return 311/6 * (g(to) - g(From))
