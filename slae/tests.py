from slae.lu_decomposer import LUDecomposer
from slae.qr_decomposer import QRDecomposer
from slae.matrix import *


def test_LU_decomposition():
    n = 4
    A = generate_random_matrix(n, n)
    decomposer = LUDecomposer(A)
    P = generate_row_permutation_matrix(A.rows, decomposer.perm_rows)
    Q = generate_column_permutation_matrix(A.cols, decomposer.perm_cols)
    L = decomposer.L
    U = decomposer.U
    LU = L * U
    PAQ = P * A * Q
    b = generate_random_matrix(n, 1)
    x = decomposer.solve(b)

    LINE_SEP = "\n" + "-" * 26 * n + "\n"
    print("=" * 20 + "  LU decomposition test  " + "=" * 20, end="\n\n")
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
    print("Ax - b = 0: " + str((A * x - b) == generate_zero_matrix(n, 1)), sep="\n", end=LINE_SEP)
    print("A^(-1):", decomposer.inverted_matrix, sep="\n", end=LINE_SEP)
    print("A * A^(-1):", A * decomposer.inverted_matrix, sep="\n")
    print("A * A^(-1) = E:", A * decomposer.inverted_matrix == generate_identity_matrix(n), end=LINE_SEP)
    print("Conditional number: ", decomposer.condition_number(), end=LINE_SEP)
    print()


def test_decomposing_singular_matrix():
    n = 4
    A = generate_random_singular_square_matrix(n)
    decomposer = LUDecomposer(A)
    P = generate_row_permutation_matrix(A.rows, decomposer.perm_rows)
    Q = generate_column_permutation_matrix(A.cols, decomposer.perm_cols)
    L = decomposer.L
    U = decomposer.U
    LU = L * U
    PAQ = P * A * Q
    b = Matrix([[1], [2], [-1], [3]])
    x = decomposer.solve(b)

    LINE_SEP = "\n" + "-" * 26 * n + "\n"
    print("=" * 20 + "  LU decomposition with singular matrix test  " + "=" * 20, end="\n\n")
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
    print("Ax - b = 0: " + str((A * x - b) == generate_zero_matrix(n, 1)), sep="\n", end=LINE_SEP)
    print()


def test_QR_decomposition():
    n = 4
    A = generate_random_matrix(n, n)
    decomposer = QRDecomposer(A)
    Q = decomposer.Q
    R = decomposer.R
    b = generate_random_matrix(n, 1)
    x = decomposer.solve(b)

    LINE_SEP = "\n" + "-" * 26 * n + "\n"
    print("=" * 20 + "  QR decomposition test  " + "=" * 20, end="\n\n")
    print("A:", A, sep="\n", end=LINE_SEP)
    print("Q:", Q, sep="\n", end=LINE_SEP)
    print("R:", R, sep="\n", end=LINE_SEP)
    print("QR:", Q * R, sep="\n", end=LINE_SEP)
    print("QR = A:", Q * R == A, end=LINE_SEP)
    print("b", b, sep="\n")
    print("Solution for b:", x, sep="\n", end=LINE_SEP)
    print("Ax - b", A * x - b, sep="\n")
    print("Ax - b = 0: " + str((A * x - b) == generate_zero_matrix(n, 1)), sep="\n", end=LINE_SEP)
    print()


if __name__ == '__main__':
    # test_LU_decomposition()
    # test_decomposing_singular_matrix()
    test_QR_decomposition()
