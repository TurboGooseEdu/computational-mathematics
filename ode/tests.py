from ode.cauchy_problem import *
import matplotlib.pyplot as plt


def render_plot_with_numerical_and_exact_solutions(x_values, Y_values):
    n = len(Y_values[0])
    y_values_num = [[] for _ in range(n)]
    for y in Y_values:
        for i in range(n):
            y_values_num[i].append(y[i])
    y_values_exact = [[] for _ in range(n)]
    for x in x_values:
        exact_sol = exact_solution(x)
        for j in range(n):
            y_values_exact[j].append(exact_sol[j])
    plt.title("Integral curves")
    plt.plot(x_values, y_values_exact[0], "r")
    plt.plot(x_values, y_values_num[0], "r--")
    plt.plot(x_values, y_values_exact[1], "g")
    plt.plot(x_values, y_values_num[1], "g--")
    plt.plot(x_values, y_values_exact[2], "b")
    plt.plot(x_values, y_values_num[2], "b--")
    plt.plot(x_values, y_values_exact[3], "y")
    plt.plot(x_values, y_values_num[3], "y--")
    plt.legend(["exact solution", "numerical solution"])
    plt.xlabel("x")
    plt.ylabel("y")
    plt.grid()
    plt.show()


def render_plot_with_error_of_x_dependency(x_values, Y_values):
    errors = [np.linalg.norm(Y_values[i] - exact_solution(x_values[i])) for i in range(len(x_values))]
    plt.title("error(x)")
    plt.plot(x_values, errors)
    plt.xlabel("x")
    plt.ylabel("error")
    plt.grid()
    plt.show()


def render_plot_with_h_of_x_dependency(x_values, x_discarded):
    i_d = 0
    i_x = 0
    h_values = []
    h_discarded = []
    x_accepted = []
    h_accepted = []
    while i_d < len(x_discarded) and i_x < len(x_values) - 1:
        h = x_values[i_x + 1] - x_values[i_x]
        h_values.append(h)
        if x_values[i_x] == x_discarded[i_d]:
            h_discarded.append(h)
            i_d += 1
        else:
            h_accepted.append(h)
            x_accepted.append(x_values[i_x])
        i_x += 1
    while i_x < len(x_values) - 1:
        h = x_values[i_x + 1] - x_values[i_x]
        h_values.append(h)
        h_accepted.append(h)
        x_accepted.append(x_values[i_x])
        i_x += 1
    plt.plot(x_values[:-1], h_values)
    plt.plot(x_accepted, h_accepted, "ro")
    plt.plot(x_discarded, h_discarded, "bx")
    plt.legend(["curve", "accepted", "discarded"])
    plt.title("h(x)")
    plt.xlabel("x")
    plt.ylabel("h")
    plt.grid()
    plt.show()


def test10():
    render_plot_with_numerical_and_exact_solutions(*solve_constant_step(x_start, x_fin, 0.01, considered_method))


def test11():
    h = 0.01
    h_values = []
    errors = []
    Yn_exact = exact_solution(x_fin)
    for i in range(5):
        h_values.append(h)
        errors.append(np.linalg.norm(solve_constant_step(x_start, x_fin, h, considered_method)[1][-1] - Yn_exact))
        h /= 2
    plt.title("error(h)")
    plt.plot(h_values, errors)
    plt.semilogy(base=2)
    plt.semilogx(base=2)
    plt.xlabel("h")
    plt.ylabel("error")
    plt.grid()
    plt.show()


def test12():
    h = 0.01
    tol = 1.e-5
    p = considered_method.precision
    Yn = solve_constant_step(x_start, x_fin, h, considered_method)[1][-1]
    Y2n = solve_constant_step(x_start, x_fin, h/2, considered_method)[1][-1]
    R2n = eval_error_by_Runge(Yn, Y2n, considered_method)
    h_opt = h / 2 * (tol / np.linalg.norm(R2n))**(1 / p)
    x_values, Y_values = solve_constant_step(x_start, x_fin, h_opt, considered_method)
    render_plot_with_error_of_x_dependency(x_values, Y_values)


def test21(x_values, Y_values):
    render_plot_with_numerical_and_exact_solutions(x_values, Y_values)


def test22(x_values, x_discarded):
    render_plot_with_h_of_x_dependency(x_values, x_discarded)


def test23(x_values, Y_values):
    render_plot_with_error_of_x_dependency(x_values, Y_values)


def test24():
    p = considered_method.precision
    tol = 1.e-4
    tol_values = []
    f_calculation_count = []
    for i in range(5):
        tol_values.append(tol)
        f_calculation_count.append(3 * p * len(solve_auto_step(x_start, x_fin, considered_method, tol)[0]))
        tol /= 10
    plt.title("f_calculations(tol)")
    plt.plot(tol_values, f_calculation_count)
    plt.semilogy(base=2)
    plt.semilogx(base=2)
    plt.xlabel("tol")
    plt.ylabel("f_calculations")
    plt.grid()
    plt.show()


def test1():
    test10()
    test11()
    test12()


def test2():
    x_values, Y_values, x_discarded = solve_auto_step(x_start, x_fin, considered_method)
    test21(x_values, Y_values)
    test22(x_values, x_discarded)
    test23(x_values, Y_values)
    test24()


if __name__ == '__main__':
    test1()
    test2()
