from math import sin, cos, exp, inf, log
from slae.matrix import Matrix
from integrals.lu_decomposer import LUDecomposer
import sympy as sp

a = 2.1
b = 3.3
alpha = 2/5
exact_answer = 4.461512705331194112840828080521604042844


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
    return A1 * f(z1) + A2 * f(z2) + A3 * f(z3)


def IQF_evaluation(From, z1, z2, z3, to):
    g = lambda x: (x - a) ** (1 - alpha) * (
                ((a - x) * (3 * a ** 2 - 2 * a * (z1 + z2 + z3) + z1 * (z2 + z3) + z2 * z3)) / (alpha - 2) - (
                    (a - x) ** 2 * (3 * a - z1 - z2 - z3)) / (alpha - 3) + (a - x) ** 3 / (alpha - 4) - (
                    (a - z1) * (a - z2) * (a - z3)) / (alpha - 1))
    return 311/6 * (g(to) - g(From))


def custom_QF(QF, k):
    h = (b - a) / k
    result = 0
    for i in range(k):
        result += QF(a + i * h, a + (i + 1) * h)
    return result


def calculate_integral(QF, eps=1.e-6):
    L = 2
    h = 1  # dividing [a, b] for h equal parts
    S = []
    error = inf
    while error > eps:
        n = len(S)
        if n >= 3:
            S1 = S[n - 3]
            S2 = S[n - 2]
            S3 = S[n - 1]
            m = log(abs((S2 - S1) / (S3 - S2)), L)

            matrix = []
            for i in range(n):
                tmp = [1]
                step = (b - a) / L ** i
                for j in range(n - 1):
                    tmp.append(step**(m + j))
                matrix.append(tmp)

            A = Matrix(matrix)
            B = Matrix([[i] for i in S])
            sol = LUDecomposer(A).solve(B)

            -J = sol[0][0]
            R = 0
            for i in range(1, n):
                R += sol[i][0] * ((b - a) / L ** (i - 1)) ** (m + i - 1)

            error = R

        S.append(custom_QF(QF, h))
        h *= L

    return S[-1]
