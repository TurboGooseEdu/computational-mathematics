from slae.lu_decomposer import LUDecomposer
from slae.qr_decomposer import QRDecomposer
from slae.iteration_methods import *
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
    print("=" * 20 + "  LU decomposition  " + "=" * 20, end="\n\n")
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
    print("=" * 20 + "  LU decomposition with singular matrix  " + "=" * 20, end="\n\n")
    print("A:", A, sep="\n")
    print("rank(A) = ", decomposer.rank, end=LINE_SEP)
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
    print("=" * 20 + "  QR decomposition  " + "=" * 20, end="\n\n")
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


def test_Jacoby_and_Seidel_methods():
    test_matrix_with_diagonal_dominance()
    test_symmetric_positive_defined_matrix()


def test_matrix_with_diagonal_dominance():
    n = 3
    LINE_SEP = "\n" + "-" * 26 * n + "\n"

    A = generate_random_matrix_with_diagonal_dominance(n)
    b = Matrix([[i + 1] for i in range(n)])
    x_exact = LUDecomposer(A).solve(b)
    B, c = decompose_Jacobi(A, b)

    print("=" * 20 + "  Matrix with diagonal dominance  " + "=" * 20, end="\n\n")
    print("A:", A, sep="\n", end=LINE_SEP)
    print("b", b, sep="\n", end=LINE_SEP)
    print("Exact solution for b (using LU decomposition):", x_exact, sep="\n", end=LINE_SEP)

    x, prior, posterior = solve_simple_iteration(B, c)
    print("Solution for b (using Jacoby method):", x, sep="\n", end=LINE_SEP)
    print("Prior evaluation: ", prior, "\nPosterior evaluation: ", posterior, end=LINE_SEP)
    print("Ax - b", A * x - b, sep="\n")
    print("Ax - b = 0: " + str((A * x - b) == generate_zero_matrix(n, 1)), sep="\n", end=LINE_SEP)
    print("x - x_exact:", x - x_exact, sep="\n", end=LINE_SEP)

    x, prior, posterior = solve_Seidel(B, c)
    print("Solution for b (using Seidel method):", x, sep="\n", end=LINE_SEP)
    print("Prior evaluation: ", prior, "\nPosterior evaluation: ", posterior, end=LINE_SEP)
    print("Ax - b", A * x - b, sep="\n")
    print("Ax - b = 0: " + str((A * x - b) == generate_zero_matrix(n, 1)), sep="\n", end=LINE_SEP)
    print("x - x_exact:", x - x_exact, sep="\n", end=LINE_SEP)
    print()


def test_symmetric_positive_defined_matrix():
    n = 3
    LINE_SEP = "\n" + "-" * 26 * n + "\n"

    A = generate_random_positive_defined_symmetric_matrix(n)
    b = Matrix([[i + 1] for i in range(n)])
    x_exact = LUDecomposer(A).solve(b)
    B, c = decompose_Jacobi(A, b)

    print("=" * 20 + "  Positive defined symmetric matrix  " + "=" * 20, end="\n\n")
    print("A:", A, sep="\n", end=LINE_SEP)
    print("b", b, sep="\n", end=LINE_SEP)
    print("Exact solution for b (using LU decomposition):", x_exact, sep="\n", end=LINE_SEP)

    x, evaluated, actual = solve_Seidel(B, c)
    print("Solution for b (using Seidel method):", x, sep="\n", end=LINE_SEP)
    print("Iterations:", actual, end=LINE_SEP)
    print("Ax - b", A * x - b, sep="\n")
    print("Ax - b = 0: " + str((A * x - b) == generate_zero_matrix(n, 1)), sep="\n", end=LINE_SEP)
    print("x - x_exact:", x - x_exact, sep="\n", end=LINE_SEP)
    print()


if __name__ == '__main__':
    # test_LU_decomposition()
    # test_decomposing_singular_matrix()
    # test_QR_decomposition()
    test_Jacoby_and_Seidel_methods()
