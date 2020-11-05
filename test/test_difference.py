from helpers import assert_norm_equality, save

import dmsh


def test_difference(show=False):
    geo = dmsh.Difference(dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0))
    X, cells = dmsh.generate(geo, 0.1, show=show)

    geo.plot()

    ref_norms = [2.9445582949135735e02, 1.5856370081862632e01, 1.4999999025187867e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-7)
    return X, cells


if __name__ == "__main__":
    X, cells = test_difference(show=True)
    save("difference.png", X, cells)
