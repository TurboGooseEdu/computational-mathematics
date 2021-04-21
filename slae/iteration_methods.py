from slae.matrix import *
from math import log, ceil


def decompose_Jacobi(A, b):
    if A.rows == A.cols and b.rows == A.rows and b.cols == 1:
        n = A.rows
        B = generate_zero_matrix(n, n)
        for i in range(n):
            for j in range(n):
                if i != j:
                    B[i][j] = -A[i][j] / A[i][i]
        c = b.copy()
        for i in range(n):
            c[i][0] /= A[i][i]
        return B, c


def solve_Seidel(B, c, EPS=1.e-16):
    if B.rows == B.cols and c.rows == B.rows and c.cols == 1:
        n = B.rows
        abs_B = abs(B)
        expected_iterations = -1
        actual_iterations = 0

        if abs_B < 1:
            abs_B_R = max(sum(abs(B[i][j]) for j in range(i + 1, n)) for i in range(n))
            expected_iterations = ceil(log(EPS*(1 - abs_B)/abs(c), abs_B))
            EPS *= (1 - abs_B) / abs_B_R
        else:
            EPS /= 100

        x_prev = generate_zero_matrix(n, 1)
        x_cur = c.copy()
        while abs(x_cur - x_prev) > EPS:
            x_prev = x_cur.copy()
            for i in range(n):
                x_sum = c[i][0]
                for j in range(n):
                    x_sum += B[i][j] * x_cur[j][0]
                x_cur[i][0] = x_sum
            actual_iterations += 1
        return x_cur, expected_iterations, actual_iterations


def solve_simple_iteration(B, c, EPS=1.e-16):
    if B.rows == B.cols and c.rows == B.rows and c.cols == 1:
        n = B.rows
        abs_B = abs(B)
        expected_iterations = -1
        actual_iterations = 0

        if abs_B < 1:
            abs_B_R = max(sum(abs(B[i][j]) for j in range(i + 1, n)) for i in range(n))
            expected_iterations = ceil(log(EPS*(1 - abs_B)/abs(c), abs_B))
            EPS *= (1 - abs_B) / abs_B_R
        else:
            EPS /= 100

        x_prev = generate_zero_matrix(n, 1)
        x_cur = c.copy()
        while abs(x_cur - x_prev) > EPS:
            x_prev = x_cur.copy()
            x_cur = B * x_cur + c
            actual_iterations += 1
        return x_cur, expected_iterations, actual_iterations
