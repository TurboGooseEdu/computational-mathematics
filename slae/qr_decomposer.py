from slae.matrix import *
from math import sqrt


class QRDecomposer:
    def __init__(self, matrix):
        self.matrix = matrix
        if matrix.rows == matrix.cols:
            self.Q, self.R = self.decompose_QR()

    def decompose_QR(self):
        R = self.matrix.copy()
        n = self.matrix.rows
        Q = generate_identity_matrix(n)
        for i in range(n):
            for j in range(i):
                sin = R[i][j] / sqrt(R[j][j]**2 + R[i][j]**2)
                cos = R[j][j] / sqrt(R[j][j]**2 + R[i][j]**2)
                for k in range(n):
                    new_a_i_k = cos * R[i][k] - sin * R[j][k]
                    new_a_j_k = cos * R[j][k] + sin * R[i][k]
                    R[i][k] = new_a_i_k
                    R[j][k] = new_a_j_k
                new_Q = generate_identity_matrix(n)
                new_Q[i][i] = cos
                new_Q[i][j] = sin
                new_Q[j][i] = -sin
                new_Q[j][j] = cos
                Q *= new_Q
        return Q, R

    def solve(self, b):
        b = self.Q.transpose() * b
        n = self.matrix.rows
        for i in reversed(range(n)):
            b[i][0] /= self.R[i][i]
            for j in reversed(range(i)):
                b[j][0] -= b[i][0] * self.R[j][i]
        return b
