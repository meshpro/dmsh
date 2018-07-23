# -*- coding: utf-8 -*-
#
import dmsh

from helpers import assert_norm_equality


def test(show=True):
    geo = dmsh.Translation(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), [1.0, 1.0])
    X, cells = dmsh.generate(geo, 0.1, show=show)

    ref_norms = [1.7525e+03, 5.5677441324948013e+01, 3.0]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-12)
    return


if __name__ == "__main__":
    test(show=False)
