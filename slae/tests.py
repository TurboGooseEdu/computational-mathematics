from slae.lu_decomposer import LUDecomposer
from slae.matrix import *


def test_LU_decomposition():
    n = 3
    A = generate_random_matrix(n, n)
    decomposer = LUDecomposer(A)
    P = generate_row_permutation_matrix(A.rows, decomposer.perm_rows)
    Q = generate_column_permutation_matrix(A.cols, decomposer.perm_cols)
    L = decomposer.L
    U = decomposer.U
    LU = L * U
    PAQ = P * A * Q
    b = Matrix([[1 + i] for i in range(n)])
    x = decomposer.solve(b)

    LINE_SEP = "\n" + "-" * 26 * n + "\n"
    print("A:", A, sep="\n", end=LINE_SEP)
    print("L:", L, sep="\n", end=LINE_SEP)
    print("U:", U, sep="\n", end=LINE_SEP)
    print("P:", P, sep="\n", end=LINE_SEP)
    print("Q:", Q, sep="\n", end=LINE_SEP)
    print("LU:", LU, sep="\n", end=LINE_SEP)
    print("PAQ:", PAQ, sep="\n", end=LINE_SEP)
    print("LU = PAQ:", LU == PAQ, end=LINE_SEP)
    print("det(A):", decomposer.det(), end=LINE_SEP)
    print("b", b, sep="\n")
    print("Solution for b:", x, sep="\n", end=LINE_SEP)
    print("Ax - b", A * x - b, sep="\n")
    print("Ax - b = 0: " + str((A * x - b) == generate_zero_matrix(3, 1)), sep="\n", end=LINE_SEP)
    print("A^(-1):", decomposer.inverted_matrix, sep="\n", end=LINE_SEP)
    print("A * A^(-1):", A * decomposer.inverted_matrix, sep="\n")
    print("A * A^(-1) = E:", A * decomposer.inverted_matrix == generate_identity_matrix(A.rows), end=LINE_SEP)
    print("Conditional number: ", decomposer.condition_number())


def test_decomposing_singular_matrix():
    n = 3
    A = generate_random_singular_square_matrix(n)
    decomposer = LUDecomposer(A)
    P = generate_row_permutation_matrix(A.rows, decomposer.perm_rows)
    Q = generate_column_permutation_matrix(A.cols, decomposer.perm_cols)
    L = decomposer.L
    U = decomposer.U
    LU = L * U
    PAQ = P * A * Q
    b = Matrix([[1 + i] for i in range(n)])
    x = decomposer.solve(b)

    LINE_SEP = "\n" + "-" * 26 * n + "\n"
    print("A:", A, sep="\n", end=LINE_SEP)
    print("L:", L, sep="\n", end=LINE_SEP)
    print("U:", U, sep="\n", end=LINE_SEP)
    print("P:", P, sep="\n", end=LINE_SEP)
    print("Q:", Q, sep="\n", end=LINE_SEP)
    print("LU:", LU, sep="\n", end=LINE_SEP)
    print("PAQ:", PAQ, sep="\n", end=LINE_SEP)
    print("LU = PAQ:", LU == PAQ, end=LINE_SEP)
    print("det(A):", decomposer.det(), end=LINE_SEP)
    print("b", b, sep="\n")
    print("Solution for b:", x, sep="\n", end=LINE_SEP)
    print("Ax - b", A * x - b, sep="\n")
    print("Ax - b = 0: " + str((A * x - b) == generate_zero_matrix(3, 1)), sep="\n", end=LINE_SEP)


if __name__ == '__main__':
    test_LU_decomposition()
    # test_decomposing_singular_matrix()
