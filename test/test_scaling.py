# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test(h0=1.8, show=True):
    geo = dmsh.Scaling(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), 2.0)
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [1, 4, 0],
            [4, 3, 0],
            [3, 4, 6],
            [5, 1, 2],
            [1, 5, 4],
            [4, 7, 6],
            [5, 7, 4],
            [7, 5, 8],
        ],
    ), cells

    assert_norm_equality(X.flatten(), [3.3e+01, 9.3310450883084606e+00, 4.0], 1.0e-12)
    return


if __name__ == "__main__":
    test(h0=0.1)
