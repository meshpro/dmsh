# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test_union(h0=1.4, show=True):
    angles = numpy.pi * numpy.array([3.0 / 6.0, 7.0 / 6.0, 11.0 / 6.0])
    geo = dmsh.Union(
        [
            dmsh.Circle([numpy.cos(angles[0]), numpy.sin(angles[0])], 1.0),
            dmsh.Circle([numpy.cos(angles[1]), numpy.sin(angles[1])], 1.0),
            dmsh.Circle([numpy.cos(angles[2]), numpy.sin(angles[2])], 1.0),
        ]
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [3, 2, 8],
            [4, 1, 2],
            [0, 4, 2],
            [4, 0, 6],
            [10, 0, 2],
            [3, 10, 2],
            [10, 9, 0],
            [9, 10, 7],
            [1, 5, 2],
            [2, 5, 8],
        ],
    ), cells

    assert_norm_equality(
        X.flatten(), [2.0113882877340394e+01, 5.2084888257243884e+00, 2.0], 1.0e-12
    )
    return


if __name__ == "__main__":
    test_union(h0=0.3)
