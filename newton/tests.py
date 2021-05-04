from slae.matrix import *
from newton.newton_method import newton_method
from math import exp, sin, cos, sinh, cosh


def F(matrix):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = [matrix[i][0] for i in range(matrix.rows)]
    return Matrix(
        [[cos(x2 * x1) - exp(-3 * x3) + x4 * x5 ** 2 - x6 - sinh(2 * x8) * x9 + 2 * x10 + 2.000433974165385440],
         [sin(x2 * x1) + x3 * x9 * x7 - exp(-x10 + x6) + 3 * x5 ** 2 - x6 * (x8 + 1) + 10.886272036407019994],
         [x1 - x2 + x3 - x4 + x5 - x6 + x7 - x8 + x9 - x10 - 3.1361904761904761904],
         [2 * cos(-x9 + x4) + x5 / (x3 + x1) - sin(x2 ** 2) + cos(x7 * x10) ** 2 - x8 - 0.1707472705022304757],
         [sin(x5) + 2 * x8 * (x3 + x1) - exp(-x7 * (-x10 + x6)) + 2 * cos(x2) - 1.0 / (-x9+x4) - 0.3685896273101277862],
         [exp(x1 - x4 - x9) + x5 ** 2 / x8 + cos(3 * x10 * x2) / 2 - x6 * x3 + 2.0491086016771875115],
         [x2 ** 3 * x7 - sin(x10 / x5 + x8) + (x1 - x6) * cos(x4) + x3 - 0.7380430076202798014],
         [x5 * (x1 - 2 * x6) ** 2 - 2 * sin(-x9 + x3) + 0.15e1 * x4 - exp(x2 * x7 + x10) + 3.5668321989693809040],
         [7 / x6 + exp(x5 + x4) - 2 * x2 * x8 * x10 * x7 + 3 * x9 - 3 * x1 - 8.4394734508383257499],
         [x10 * x1 + x9 * x2 - x8 * x3 + sin(x4 + x5 + x6) * x7 - 0.78238095238095238096]])


def J(matrix):
    x1, x2, x3, x4, x5, x6, x7, x8, x9, x10 = [matrix[i][0] for i in range(matrix.rows)]
    return Matrix(
        [[-x2 * sin(x2 * x1), -x1 * sin(x2 * x1), 3 * exp(-3 * x3), x5 ** 2, 2 * x4 * x5,
          -1, 0, -2 * cosh(2 * x8) * x9, -sinh(2 * x8), 2],
         [x2 * cos(x2 * x1), x1 * cos(x2 * x1), x9 * x7, 0, 6 * x5,
          -exp(-x10 + x6) - x8 - 1, x3 * x9, -x6, x3 * x7, exp(-x10 + x6)],
         [1, -1, 1, -1, 1, -1, 1, -1, 1, -1],
         [-x5 / (x3 + x1) ** 2, -2 * x2 * cos(x2 ** 2), -x5 / (x3 + x1) ** 2, -2 * sin(-x9 + x4),
          1.0 / (x3 + x1), 0, -2 * cos(x7 * x10) * x10 * sin(x7 * x10), -1,
          2 * sin(-x9 + x4), -2 * cos(x7 * x10) * x7 * sin(x7 * x10)],
         [2 * x8, -2 * sin(x2), 2 * x8, 1.0 / (-x9 + x4) ** 2, cos(x5),
          x7 * exp(-x7 * (-x10 + x6)), -(x10 - x6) * exp(-x7 * (-x10 + x6)), 2 * x3 + 2 * x1,
          -1.0 / (-x9 + x4) ** 2, -x7 * exp(-x7 * (-x10 + x6))],
         [exp(x1 - x4 - x9), -1.5 * x10 * sin(3 * x10 * x2), -x6, -exp(x1 - x4 - x9),
          2 * x5 / x8, -x3, 0, -x5 ** 2 / x8 ** 2, -exp(x1 - x4 - x9), -1.5 * x2 * sin(3 * x10 * x2)],
         [cos(x4), 3 * x2 ** 2 * x7, 1, -(x1 - x6) * sin(x4), x10 / x5 ** 2 * cos(x10 / x5 + x8),
          -cos(x4), x2 ** 3, -cos(x10 / x5 + x8), 0, -1.0 / x5 * cos(x10 / x5 + x8)],
         [2 * x5 * (x1 - 2 * x6), -x7 * exp(x2 * x7 + x10), -2 * cos(-x9 + x3), 1.5,
          (x1 - 2 * x6) ** 2, -4 * x5 * (x1 - 2 * x6), -x2 * exp(x2 * x7 + x10), 0, 2 * cos(-x9 + x3),
          -exp(x2 * x7 + x10)],
         [-3, -2 * x8 * x10 * x7, 0, exp(x5 + x4), exp(x5 + x4),
          -7.0 / x6 ** 2, -2 * x2 * x8 * x10, -2 * x2 * x10 * x7, 3, -2 * x2 * x8 * x7],
         [x10, x9, -x8, cos(x4 + x5 + x6) * x7, cos(x4 + x5 + x6) * x7,
          cos(x4 + x5 + x6) * x7, sin(x4 + x5 + x6), -x3, x2, x1]])


X0 = Matrix([[0.5], [0.5], [1.5], [-1], [-0.5], [1.5], [0.5], [-0.5], [1.5], [-1.5]])


def test_case(k, m):
    x, iterations, operations = newton_method(F, J, X0, k=k, m=m)
    LINE_SEP = "\n" + "-" * 30 + "\n\n"
    print("Solution x:", x, sep="\n", end=LINE_SEP)
    print("Iterations: ", iterations, "\nOperations: ", operations, end=LINE_SEP)
    print("F(X):", F(x), sep="\n")
    print("F(X) = 0:", F(x) == generate_zero_matrix(10, 1), end="\n\n")


def test_ordinary_nm():
    test_case(-1, 1)


def test_modified_nm():
    test_case(1, -1)


def test_combined_nm():
    test_case(8, 3)


if __name__ == '__main__':
    print("-" * 15 + "  Ordinary NM  " + "-" * 15 + "\n")
    test_ordinary_nm()
    print("-" * 15 + "  Modified NM  " + "-" * 15 + "\n")
    test_modified_nm()
    print("-" * 15 + "  Combined NM  " + "-" * 15 + "\n")
    test_combined_nm()
