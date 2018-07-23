# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test_rectangle(h0=0.8, show=True):
    geo = dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0)
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [8, 7, 0],
            [3, 7, 8],
            [4, 8, 0],
            [10, 3, 8],
            [9, 4, 5],
            [4, 9, 8],
            [10, 9, 11],
            [9, 10, 8],
            [6, 5, 1],
            [6, 9, 5],
            [11, 12, 2],
            [9, 12, 11],
            [6, 12, 9],
        ],
    ), cells

    assert_norm_equality(
        X.flatten(), [2.2613990510663321e+01, 5.4775659192375228e+00, 2.0], 1.0e-12
    )
    return


if __name__ == "__main__":
    test_rectangle(0.1)
