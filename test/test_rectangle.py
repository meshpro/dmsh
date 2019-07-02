import dmsh
from helpers import assert_norm_equality, save


def test_rectangle(show=False):
    geo = dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0)
    X, cells = dmsh.generate(geo, 0.1, show=show)

    ref_norms = [9.7542898028694776e02, 3.1710503119308623e01, 2.0]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-12)
    return X, cells


if __name__ == "__main__":
    X, cells = test_rectangle(show=False)
    save("rectangle.png", X, cells)
