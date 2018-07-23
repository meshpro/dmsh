# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test_difference(h0=0.5, show=True):
    geo = dmsh.Difference(dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0))
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [0, 5, 3],
            [5, 2, 3],
            [9, 7, 5],
            [4, 7, 6],
            [10, 7, 9],
            [7, 2, 5],
            [2, 7, 4],
            [11, 9, 1],
            [11, 10, 9],
            [7, 8, 6],
            [8, 7, 10],
        ],
    ), cells

    assert_norm_equality(
        X.flatten(), [1.6199223810300637e+01, 3.9266705182833972e+00, 1.5], 1.0e-12
    )
    return


if __name__ == "__main__":
    test_difference(0.1, show=True)
