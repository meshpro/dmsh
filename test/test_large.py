from helpers import assert_norm_equality

import dmsh


def test_large():
    # https://github.com/nschloe/dmsh/issues/11
    r = dmsh.Rectangle(-10.0, +20.0, -10.0, +20.0)
    c = dmsh.Circle([0.0, 0.0], 3)
    geo = dmsh.Difference(r, c)

    X, cells = dmsh.generate(geo, 2.0, tol=1.0e-5, max_steps=10000)

    ref_norms = [4.6422985179724637e03, 2.4202897690682192e02, 2.0000000000000000e01]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-4)
