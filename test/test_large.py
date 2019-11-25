import dmsh
from helpers import assert_norm_equality


def test_large():
    # https://github.com/nschloe/dmsh/issues/11
    r = dmsh.Rectangle(-10.0, +20.0, -10.0, +20.0)
    c = dmsh.Circle([0.0, 0.0], 3)
    geo = dmsh.Difference(r, c)

    X, cells = dmsh.generate(geo, 2.0, tol=1.0e-5, max_steps=10000)

    ref_norms = [4.4714910237327376e+03, 2.3785742555298563e+02, 2.0000000000000000e+01]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-12)
