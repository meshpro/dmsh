# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test_halfspace(h0=0.4, show=True):
    geo = dmsh.Intersection(
        [
            dmsh.HalfSpace(numpy.sqrt(0.5) * numpy.array([1.0, 1.0]), 0.0),
            dmsh.Circle([0.0, 0.0], 1.0),
        ]
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [3, 2, 1],
            [10, 11, 0],
            [5, 3, 6],
            [3, 5, 2],
            [2, 5, 4],
            [5, 8, 4],
            [11, 7, 12],
            [7, 11, 10],
            [7, 13, 12],
            [13, 7, 8],
            [7, 10, 4],
            [8, 7, 4],
            [9, 13, 8],
            [9, 5, 6],
            [5, 9, 8],
        ],
    )

    assert_norm_equality(
        X.flatten(),
        [1.3905252276781415e+01, 3.0919890978059588e+00, 9.9320349979775846e-01],
        1.0e-12,
    )
    return


if __name__ == "__main__":
    test_halfspace(h0=0.1, show=True)
