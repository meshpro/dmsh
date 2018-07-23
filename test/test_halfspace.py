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

    print(cells)
    assert numpy.array_equal(
        cells,
        [
            [2, 5, 4],
            [3, 2, 1],
            [5, 3, 6],
            [3, 5, 2],
            [10, 11, 0],
            [5, 7, 4],
            [9, 5, 6],
            [9, 13, 5],
            [7, 8, 10],
            [11, 8, 12],
            [8, 11, 10],
            [8, 7, 5],
            [8, 13, 12],
            [13, 8, 5],
        ],
    ), cells

    assert_norm_equality(
        X.flatten(),
        [1.3840082952648441e+01, 3.0538023439033259e+00, 9.9346459023302125e-01],
        1.0e-12,
    )
    return


if __name__ == "__main__":
    test_halfspace(h0=0.1, show=True)
