from helpers import assert_norm_equality, save

import dmsh


def test_ellipse(show=False):
    geo = dmsh.Ellipse([0.0, 0.0], 2.0, 1.0)
    X, cells = dmsh.generate(geo, 0.2, show=show)

    geo.plot()

    ref_norms = [2.5108941453435716e02, 1.5652963447587933e01, 1.9890264390440919e00]
    assert_norm_equality(X.flatten(), ref_norms, 2.0e-2)
    return X, cells


if __name__ == "__main__":
    X, cells = test_ellipse(show=True)
    save("ellipse.png", X, cells)
