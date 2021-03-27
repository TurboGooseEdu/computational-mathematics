from slae.lu_decomposer import LUDecomposer
from slae.matrix import *


if __name__ == '__main__':
    A = generate_random_matrix(3, 3)
    decomposer = LUDecomposer(A)
    P = generate_permutation_matrix(A.rows, decomposer.permutations)
    L = decomposer.L
    U = decomposer.U
    M = decomposer.LU
    LU = L * U
    PA = P * A

    LINE_SEP = "\n" + "-" * 80 + "\n"
    print("A:", A, sep="\n", end=LINE_SEP)
    print("L:", L, sep="\n", end=LINE_SEP)
    print("U:", U, sep="\n", end=LINE_SEP)
    print("P:", P, sep="\n", end=LINE_SEP)
    print("LU:", LU, sep="\n", end=LINE_SEP)
    print("PA:", PA, sep="\n", end=LINE_SEP)
    print("LU == PA:", LU == PA, end=LINE_SEP)
    print("det(A):", decomposer.det(), end=LINE_SEP)
    print("Solution for b = [0, 0, 1]:", decomposer.solve([0, 0, 1]), sep="\n", end=LINE_SEP)
    print("A^(-1):", decomposer.inverted_matrix, sep="\n", end=LINE_SEP)
    print("A * A^(-1) = E:", A * decomposer.inverted_matrix == generate_identity_matrix(A.rows), end=LINE_SEP)
    print("Conditional number: ", decomposer.condition_number())
