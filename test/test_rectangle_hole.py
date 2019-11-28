import dmsh
from helpers import assert_norm_equality


def test_rectangle_hole():
    r = dmsh.Rectangle(60, 330, 380, 650)
    h = dmsh.Rectangle(143, 245, 440, 543)
    geo = dmsh.Difference(r, h)

    X, cells = dmsh.generate(geo, 20, tol=1.0e-5, show=False)

    ref_norms = [1.2375342190668483e05, 7.4629069204951311e03, 6.5e02]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-12)
