from helpers import assert_norm_equality

import dmsh


def test_rectangle_hole(show=False):
    geo = dmsh.Difference(
        dmsh.Rectangle(60, 330, 380, 650), dmsh.Rectangle(143, 245, 440, 543)
    )

    X, cells = dmsh.generate(geo, 20, tol=1.0e-5, show=show, flip_tol=1.0e-10)

    ref_norms = [1.2895976881893826e05, 7.6183307832433229e03, 6.5000000000000000e02]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)


def test_rectangle_hole2(show=False):
    geo = dmsh.Difference(
        dmsh.Rectangle(0.0, 5.0, 0.0, 5.0),
        dmsh.Polygon([[1, 1], [4, 1], [4, 4], [1, 4]]),
    )
    X, cells = dmsh.generate(geo, 1.0, show=show, tol=1.0e-3)

    ref_norms = [1.4367314502085827e02, 2.3325977831484149e01, 5.0000000000000000e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)


if __name__ == "__main__":
    test_rectangle_hole2(show=True)
