from slae.matrix import *


class LUDecomposer:
    def __init__(self, matrix):
        self.matrix = matrix
        self.LU, self.perm_rows, self.perm_cols = self.decompose_LU()

    def decompose_LU(self):
        matrix = self.matrix.copy()
        EPS = abs(matrix) * 1e-15
        if matrix.rows == matrix.cols:
            n = matrix.rows
            perm_rows = []
            perm_cols = []
            for i in range(n):
                max_element = 0
                max_elem_row = -1
                max_elem_col = -1
                for j in range(i, n):
                    for k in range(i, n):
                        elem = matrix[j][k]
                        if abs(elem) > EPS and abs(elem) > abs(max_element):
                            max_element = elem
                            max_elem_row = j
                            max_elem_col = k
                if max_elem_row == -1 or max_elem_col == -1:
                    break
                if max_elem_row != i:
                    matrix.switch_rows(i, max_elem_row)
                    perm_rows.append((i, max_elem_row))
                if max_elem_col != i:
                    matrix.switch_cols(i, max_elem_col)
                    perm_cols.append((i, max_elem_col))
                for j in range(i + 1, n):
                    elem = matrix[j][i]
                    if abs(elem) < EPS:
                        continue
                    for k in range(i, n):
                        matrix[j][k] -= (matrix[i][k] * elem / matrix[i][i])
                    matrix[j][i] = elem / matrix[i][i]
            return matrix, perm_rows, perm_cols

    def solve(self, b):
        b = b.copy()
        n = self.LU.rows
        for p in self.perm_rows:
            b.switch_rows(*p)
        for i in range(n - 1):
            for j in range(i + 1, n):
                b[j][0] -= b[i][0] * self.LU[j][i]
        for i in reversed(range(n)):
            b[i][0] /= self.LU[i][i]
            for j in reversed(range(i)):
                b[j][0] -= b[i][0] * self.LU[j][i]
        return generate_column_permutation_matrix(n, self.perm_cols) * b
