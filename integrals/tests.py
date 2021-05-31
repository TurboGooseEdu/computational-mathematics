from integrals.integral import *


def test_NK():
    print("-"*40, "  Newton-Kotes  ", "-"*40)
    calculate_integral_newton_kotes()
    print()


def test_G():
    print("-"*40, "  Gauss  ", "-"*40)
    calculate_integral_gauss()
    print()


if __name__ == '__main__':
    pd.set_option('display.max_rows', 500)
    pd.set_option('display.max_columns', 500)
    pd.set_option('display.width', 1000)
    test_NK()
    test_G()
