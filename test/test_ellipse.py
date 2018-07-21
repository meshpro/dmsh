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
            [12, 7, 8],
            [7, 3, 8],
            [5, 9, 4],
            [9, 5, 10],
            [12, 11, 7],
            [0, 5, 4],
            [5, 0, 1],
            [2, 3, 7],
            [6, 5, 1],
            [2, 6, 1],
            [6, 2, 7],
            [5, 6, 10],
            [6, 11, 10],
            [11, 6, 7],
        ],
    )

    assert_norm_equality(
        X.flatten(),
        [1.7698621425909550e+01, 4.3034771316326301e+00, 1.6798867227554060e+00],
        1.0e-12,
    )
    return
