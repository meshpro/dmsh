# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test_union(h0=0.9, show=True):
    geo = dmsh.Union([dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0)])
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [10, 6, 7],
            [6, 4, 7],
            [1, 6, 10],
            [0, 4, 6],
            [3, 0, 6],
            [2, 3, 5],
            [3, 2, 0],
            [1, 9, 6],
            [9, 3, 6],
            [8, 9, 1],
            [9, 8, 5],
            [3, 9, 5],
        ],
    )

    assert_norm_equality(
        X.flatten(),
        [1.3905252276781415e+01, 3.0919890978059588e+00, 9.9320349979775846e-01],
        1.0e-12,
    )
    return


if __name__ == "__main__":
    test_union(h0=0.3)
