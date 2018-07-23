# -*- coding: utf-8 -*-
#
import dmsh

from helpers import assert_norm_equality


def test(h0=1.0, show=False):
    geo = dmsh.Difference(
        dmsh.Rectangle(0.0, 5.0, 0.0, 5.0),
        dmsh.Polygon([[1, 1], [4, 1], [4, 4], [1, 4]]),
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    assert_norm_equality(
        X.flatten(), [1.3188950271580018e+02, 2.2566129442609228e+01, 5.0], 1.0e-12
    )
    return


if __name__ == "__main__":
    test(h0=0.3, show=True)
