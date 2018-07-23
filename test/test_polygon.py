# -*- coding: utf-8 -*-
#
import dmsh

from helpers import assert_norm_equality, save


def test(show=True):
    geo = dmsh.Polygon(
        [
            [0.0, 0.0],
            [1.1, 0.0],
            [1.2, 0.5],
            [0.7, 0.6],
            [2.0, 1.0],
            [1.0, 2.0],
            [0.5, 1.5],
        ]
    )
    X, cells = dmsh.generate(geo, 0.1, show=show)

    ref_norms = [4.1468030858462305e+02, 2.1861920662017866e+01, 2.0]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-12)
    return X, cells


if __name__ == "__main__":
    X, cells = test(show=False)
    save("polygon.png", X, cells)
