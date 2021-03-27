from random import randint


def generate_zero_matrix(rows, cols):
    if rows > 0 and cols > 0:
        return Matrix([[0 for _ in range(cols)] for _ in range(rows)])


def generate_identity_matrix(n):
    if n > 0:
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])


def generate_permutation_matrix(n, permutations):
    if n > 0:
        matrix = generate_identity_matrix(n)
        for p in permutations:
            matrix.switch_rows(*p)
        return matrix


def generate_random_matrix(rows, cols):
    if rows > 0 and cols > 0:
        matrix = generate_zero_matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                matrix[i][j] = randint(1, 99)
        return matrix


class Matrix:
    def __init__(self, matrix):
        self.matrix = matrix
        self.rows = len(matrix)
        self.cols = len(matrix[0])

    def __add__(self, other):
        if other.rows == self.rows and other.cols == self.cols:
            new_matrix = generate_zero_matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    new_matrix[i][j] = self.matrix[i][j] + other.matrix[i][j]
            return new_matrix

    def __sub__(self, other):
        if other.rows == self.rows and other.cols == self.cols:
            new_matrix = generate_zero_matrix(self.rows, self.cols)
            for i in range(self.rows):
                for j in range(self.cols):
                    new_matrix[i][j] = self.matrix[i][j] - other.matrix[i][j]
            return new_matrix

    def __mul__(self, other):
        if self.cols == other.rows:
            new_matrix = generate_zero_matrix(self.rows, other.cols)
            for i in range(self.rows):
                for j in range(other.cols):
                    elem = 0
                    for k in range(self.cols):
                        elem += self.matrix[i][k] * other.matrix[k][j]
                    new_matrix[i][j] = elem
            return new_matrix

    def __getitem__(self, key):
        return self.matrix[key]

    def __eq__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            return False
        result = True
        for i in range(self.rows):
            for j in range(self.cols):
                result = result and self.matrix[i][j] == self.matrix[i][j]
                if not result:
                    return False
        return result

    def __abs__(self):
        max_row = 0
        for i in range(self.rows):
            row_sum = 0
            for j in range(self.cols):
                row_sum += abs(self.matrix[i][j])
            if row_sum > max_row:
                max_row = row_sum
        return max_row

    def __str__(self):
        result = ""
        for i in range(self.rows):
            result += "[ "
            for j in range(self.cols):
                result += str(self.matrix[i][j]).ljust(25)
            result += "]\n"
        return result

    def transpose(self):
        matrix = self.copy()
        for i in range(self.rows):
            for j in range(self.cols):
                if i == j:
                    break
                matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
        return matrix

    def switch_rows(self, row1, row2):
        self.matrix[row1], self.matrix[row2] = self.matrix[row2], self.matrix[row1]

    def copy(self):
        return Matrix([[self.matrix[i][j] for j in range(self.cols)] for i in range(self.rows)])
