import dmsh

from helpers import assert_norm_equality, save


def test_difference(show=False):
    geo = dmsh.Difference(dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0))
    X, cells = dmsh.generate(geo, 0.1, show=show)

    ref_norms = [2.9445552442961758e02, 1.5856356670813716e01, 1.4999999157880513e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-9)
    return X, cells


if __name__ == "__main__":
    X, cells = test_difference(show=False)
    save("difference.png", X, cells)
