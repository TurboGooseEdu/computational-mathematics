from slae.matrix import *


class LUDecomposer:
    def __init__(self, matrix):
        self.matrix = matrix
        if matrix.rows == matrix.cols:
            self.LU, self.perm_rows, self.perm_cols, self.rank = self.decompose_LU()
            self.L = self.extract_L()
            self.U = self.extract_U()
            if self.rank == matrix.rows:
                self.inverted_matrix = self.inverse_matrix()

    def decompose_LU(self):
        matrix = self.matrix.copy()
        EPS = abs(matrix) * 1e-15
        if matrix.rows == matrix.cols:
            n = matrix.rows
            rank = 0
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
                rank += 1

            return matrix, perm_rows, perm_cols, rank

    def extract_L(self):
        result = generate_identity_matrix(self.LU.rows)
        for i in range(1, self.LU.rows):
            for j in range(self.LU.cols):
                if i == j:
                    break
                result[i][j] = self.LU[i][j]
        return result

    def extract_U(self):
        result = generate_zero_matrix(self.LU.rows, self.LU.cols)
        for i in range(self.LU.rows):
            for j in range(i, self.LU.cols):
                result[i][j] = self.LU[i][j]
        return result

    def det(self):
        det = (-1) ** len(self.perm_rows)
        for i in range(self.LU.rows):
            det *= self.LU[i][i]
        return det

    def solve(self, b):
        EPS = abs(self.matrix) * 1.e-14
        b = b.copy()
        n = self.LU.rows
        for p in self.perm_rows:
            b.switch_rows(*p)

        for i in range(n - 1):
            for j in range(i + 1, n):
                b[j][0] -= b[i][0] * self.LU[j][i]

        if all(abs(b[i][0]) < EPS for i in range(self.rank + 1, n)):
            for i in reversed(range(min(n, self.rank))):
                b[i][0] /= self.LU[i][i]
                for j in reversed(range(i)):
                    b[j][0] -= b[i][0] * self.LU[j][i]

            return generate_column_permutation_matrix(n, self.perm_cols) * b

    def inverse_matrix(self):
        n = self.matrix.rows
        solutions = [self.solve(Matrix([[1] if i == j else [0] for j in range(n)])) for i in range(n)]
        result = generate_zero_matrix(n, n)
        for i in range(len(solutions)):
            for j in range(n):
                result[i][j] = solutions[i][j][0]
        return result.transpose()

    def condition_number(self):
        return abs(self.matrix) * abs(self.inverted_matrix)
