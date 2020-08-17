from helpers import assert_norm_equality

import dmsh


def test(show=False):
    geo = dmsh.Difference(
        dmsh.Rectangle(0.0, 5.0, 0.0, 5.0),
        dmsh.Polygon([[1, 1], [4, 1], [4, 4], [1, 4]]),
    )
    X, cells = dmsh.generate(geo, 1.0, show=show, tol=1.0e-3)

    ref_norms = [1.4000000000000000e02, 2.3176757306973560e01, 5.0000000000000000e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)


if __name__ == "__main__":
    test(show=True)
