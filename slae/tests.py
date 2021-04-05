from slae.lu_decomposer import LUDecomposer
from slae.matrix import *


def test_LU_decomposition():
    A = generate_random_matrix(3, 3)
    decomposer = LUDecomposer(A)
    P = generate_row_permutation_matrix(A.rows, decomposer.perm_rows)
    Q = generate_column_permutation_matrix(A.cols, decomposer.perm_cols)
    L = decomposer.L
    U = decomposer.U
    LU = L * U
    PAQ = P * A * Q

    b = Matrix([[0], [0], [1]])
    x = decomposer.solve(b)

    LINE_SEP = "\n" + "-" * 80 + "\n"
    print("A:", A, sep="\n", end=LINE_SEP)
    print("L:", L, sep="\n", end=LINE_SEP)
    print("U:", U, sep="\n", end=LINE_SEP)
    print("P:", P, sep="\n", end=LINE_SEP)
    print("Q:", Q, sep="\n", end=LINE_SEP)
    print("LU:", LU, sep="\n", end=LINE_SEP)
    print("PAQ:", PAQ, sep="\n", end=LINE_SEP)
    print("LU = PAQ:", LU == PAQ, end=LINE_SEP)
    print("det(A):", decomposer.det(), end=LINE_SEP)
    print("Solution for b = [0, 0, 1]:", x, "Ax - b = 0: " + str(A * x - b == generate_zero_matrix(3, 1)),  sep="\n", end=LINE_SEP)
    print("A^(-1):", decomposer.inverted_matrix, sep="\n", end=LINE_SEP)
    print("A * A^(-1) = E:", A * decomposer.inverted_matrix == generate_identity_matrix(A.rows), end=LINE_SEP)
    print("Conditional number: ", decomposer.condition_number())


if __name__ == '__main__':
    test_LU_decomposition()
