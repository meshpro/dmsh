from helpers import assert_norm_equality

import dmsh


def test_rectangle_hole():
    r = dmsh.Rectangle(60, 330, 380, 650)
    h = dmsh.Rectangle(143, 245, 440, 543)
    geo = dmsh.Difference(r, h)

    X, cells = dmsh.generate(geo, 20, tol=1.0e-5, show=False)

    ref_norms = [1.2933514312764432e05, 7.6376783845909986e03, 6.5000000000000000e02]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
