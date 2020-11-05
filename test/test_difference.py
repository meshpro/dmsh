from helpers import assert_norm_equality, save

import dmsh


def test_difference(show=False):
    """
    Test the difference between two datasets.

    Args:
        show: (bool): write your description
    """
    geo = dmsh.Difference(dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0))
    X, cells = dmsh.generate(geo, 0.1, show=show)

    geo.plot()

    ref_norms = [2.9445624536736682e02, 1.5856393245241872e01, 1.4999998887792523e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-7)
    return X, cells


if __name__ == "__main__":
    X, cells = test_difference(show=True)
    save("difference.png", X, cells)
