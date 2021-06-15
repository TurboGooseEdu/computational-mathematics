from ode.cauchy_problem import *
import matplotlib.pyplot as plt


def render_plot_with_numerical_and_exact_solutions(solution):
    x_values, Y = solution
    n = len(Y[0])
    y_values_num = [[] for _ in range(n)]
    for y in Y:
        for i in range(n):
            y_values_num[i].append(y[i])

    y_values_exact = [[] for _ in range(n)]
    exact_solutions = [y1, y2, y3, y4]
    for i in range(len(x_values)):
        for j in range(n):
            y_values_exact[j].append(exact_solutions[j](x_values[i]))

    colors = ["r", "g", "b", "y"]
    for i in range(n):
        plt.plot(x_values, y_values_exact[i], colors[i % n])
        plt.plot(x_values, y_values_num[i], colors[i] + "--")
    plt.grid()
    plt.show()


def test():
    res = solve_constant_step(considered_method, 0.005)
    render_plot_with_numerical_and_exact_solutions(res)


if __name__ == '__main__':
    test()

