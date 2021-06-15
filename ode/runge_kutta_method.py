class RungeKuttaMethod:
    def __init__(self, C, A, B, precision):
        self.precision = precision
        self.C = C
        self.A = A
        self.B = B

    def calculate(self, f, h, x0, Y0):
        s = len(self.B)
        k = []
        for i in range(s):
            x = x0 + self.C[i] * h
            Y = Y0 + h * sum(k[j] * self.A[i][j] for j in range(i))
            k.append(f(x, Y))
        return Y0 + h * sum(k[i] * self.B[i] for i in range(s))
