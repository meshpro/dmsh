# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test(h0=0.8, show=True):
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
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [7, 3, 0],
            [7, 1, 3],
            [9, 8, 4],
            [8, 9, 5],
            [6, 8, 5],
            [8, 6, 3],
            [3, 6, 0],
            [8, 2, 4],
            [1, 2, 3],
            [2, 8, 3],
        ],
    ), cells

    assert_norm_equality(
        X.flatten(), [1.7852872484930099e+01, 4.8521496373584130e+00, 2.0], 1.0e-12
    )
    return


if __name__ == "__main__":
    test(h0=0.2)
