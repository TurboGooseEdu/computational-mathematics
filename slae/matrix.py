from random import randint

RAND_MIN = 1
RAND_MAX = 99


def generate_zero_matrix(rows, cols):
    if rows > 0 and cols > 0:
        return Matrix([[0 for _ in range(cols)] for _ in range(rows)])


def generate_identity_matrix(n):
    if n > 0:
        return Matrix([[1 if i == j else 0 for j in range(n)] for i in range(n)])


def generate_row_permutation_matrix(n, row_perms):
    if n > 0:
        matrix = generate_identity_matrix(n)
        for p in row_perms:
            matrix.switch_rows(*p)
        return matrix


def generate_column_permutation_matrix(n, col_perms):
    if n > 0:
        matrix = generate_identity_matrix(n)
        for p in col_perms:
            matrix.switch_cols(*p)
        return matrix


def generate_random_matrix(rows, cols):
    if rows > 0 and cols > 0:
        matrix = generate_zero_matrix(rows, cols)
        for i in range(rows):
            for j in range(cols):
                matrix[i][j] = randint(RAND_MIN, RAND_MAX)
        return matrix


def generate_random_singular_square_matrix(n):
    if n == 1:
        return Matrix([[randint(RAND_MIN, RAND_MAX)]])

    elif n == 2:
        a1 = randint(RAND_MIN, RAND_MAX)
        a2 = randint(RAND_MIN, RAND_MAX)
        k = randint(-10, 10)
        return Matrix([[a1, a2], [a1 * k, a2 * k]])

    else:
        matrix = generate_zero_matrix(n, n)
        for i in range(n - 1):
            for j in range(n):
                matrix[i][j] = randint(RAND_MIN, RAND_MAX)
        matrix[n - 1] = [matrix[0][i] + matrix[1][i] for i in range(n)]
        if n > 3:
            matrix[n - 2] = [matrix[0][i] - matrix[1][i] for i in range(n)]
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

    def __setitem__(self, key, value):
        self.matrix[key] = value

    def __eq__(self, other):
        EPS = 5 * 1e-14
        if self.rows != other.rows or self.cols != other.cols:
            return False
        for i in range(self.rows):
            for j in range(self.cols):
                if abs(self.matrix[i][j] - other.matrix[i][j]) > EPS:
                    return False
        return True

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
        return Matrix([[self.matrix[j][i] for j in range(self.rows)] for i in range(self.cols)])

    def switch_rows(self, row1, row2):
        self.matrix[row1], self.matrix[row2] = self.matrix[row2], self.matrix[row1]

    def switch_cols(self, col1, col2):
        for i in range(self.rows):
            self.matrix[i][col1], self.matrix[i][col2] = self.matrix[i][col2], self.matrix[i][col1]

    def copy(self):
        return Matrix([[self.matrix[i][j] for j in range(self.cols)] for i in range(self.rows)])
