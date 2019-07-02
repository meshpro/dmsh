import dmsh

from helpers import assert_norm_equality, save


def test(show=False):
    geo = dmsh.Scaling(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), 2.0)
    X, cells = dmsh.generate(geo, 0.1, show=show, tol=1.0e-5)

    ref_norms = [7.7120645429243405e03, 1.2509238632152577e02, 4.0]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-12)
    return X, cells


if __name__ == "__main__":
    X, cells = test(show=False)
    save("scaling.png", X, cells)
