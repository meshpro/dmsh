import dmsh
from helpers import assert_norm_equality, save


def test_ellipse(show=False):
    geo = dmsh.Ellipse([0.0, 0.0], 2.0, 1.0)
    X, cells = dmsh.generate(geo, 0.2, show=show)

    ref_norms = [2.5232677959803675e02, 1.5694955522131542e01, 1.9877107181696752e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-12)
    return X, cells


if __name__ == "__main__":
    X, cells = test_ellipse(show=False)
    save("ellipse.png", X, cells)
