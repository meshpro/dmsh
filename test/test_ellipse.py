# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test_ellipse(h0=0.7, show=True):
    geo = dmsh.Ellipse([0.0, 0.0], 2.0, 1.0)
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [9, 5, 10],
            [5, 0, 1],
            [5, 9, 4],
            [0, 5, 4],
            [12, 11, 7],
            [12, 7, 8],
            [2, 3, 7],
            [7, 3, 8],
            [11, 6, 7],
            [6, 11, 10],
            [5, 6, 10],
            [2, 6, 1],
            [6, 2, 7],
            [6, 5, 1],
        ],
    ), cells

    assert_norm_equality(
        X.flatten(), [1.9273675392415075e+01, 4.7336334768604962e+00, 2.0], 1.0e-12
    )
    return
