from newton.lu_decomposer import LUDecomposer
from slae.matrix import Matrix
from math import inf


# k - switch to modified NM after first k steps
# m - calculate J(X) every m steps

# Ordinary NM: k = -1, m = 1
# Modified NM: k = 1, m = -1
# Combined NM: k = k > 0, m = m > 0

def newton_method(F, J, x0, k=-1, m=1, EPS=1.e-15):
    iterations = 0
    operations = 0
    m_calc = 0
    x_prev = Matrix([[inf] for _ in range(x0.rows)])
    x_cur = x0.copy()
    LU = LUDecomposer(J(x0))
    operations += LU.operations
    while abs(x_cur - x_prev) > EPS:
        x_prev = x_cur
        solution, ops = LU.solve(-F(x_cur))
        operations += ops
        x_cur += solution

        if m_calc == m:
            m_calc = 0
            LU = LUDecomposer(J(x_cur))
            operations += LU.operations

        iterations += 1
        if iterations == k:
            m = -1

        m_calc += 1

    return x_cur, iterations, operations
