from slae.matrix import *


class LUDecomposer:
    def __init__(self, matrix):
        self.matrix = matrix
        self.LU, self.permutations = self.decompose_LU()
        self.L = self.extract_L()
        self.U = self.extract_U()
        self.inverted_matrix = self.inverse_matrix()

    def decompose_LU(self):
        matrix = self.matrix.copy()
        if matrix.rows == matrix.cols:
            n = matrix.rows
            permutations = []
            for i in range(n - 1):
                # Selecting pivot by column
                max_element = 0
                element_col = -1
                for j in range(i, n):
                    elem = matrix[j][i]
                    if elem != 0 and abs(elem) > abs(max_element):
                        max_element = elem
                        element_col = j
                if element_col == -1:
                    return None
                if element_col != i:
                    permutations.append((i, element_col))
                    matrix.switch_rows(i, element_col)

                for j in range(i + 1, n):
                    elem = matrix[j][i]
                    if elem == 0:
                        continue
                    for k in range(i, n):
                        matrix[j][k] -= (matrix[i][k] * elem / matrix[i][i])
                    matrix[j][i] = elem / matrix[i][i]

            return matrix, permutations

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
        det = (-1) ** len(self.permutations)
        for i in range(self.LU.rows):
            det *= self.LU[i][i]
        return det

    def solve(self, b):    # b - array of numbers
        b = [i for i in b]
        n = self.LU.rows
        for p in self.permutations:
            r1, r2 = p
            b[r1], b[r2] = b[r2], b[r1]

        for i in range(n - 1):
            for j in range(i + 1, n):
                b[j] -= b[i] * self.LU[j][i]

        for i in reversed(range(n)):
            b[i] /= self.LU[i][i]
            for j in reversed(range(i)):
                b[j] -= b[i] * self.LU[j][i]
        return b

    def inverse_matrix(self):
        n = self.matrix.rows
        return Matrix([self.solve([1 if i == j else 0 for j in range(n)]) for i in range(n)]).transpose()

    def condition_number(self):
        return abs(self.matrix) * abs(self.inverted_matrix)
