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
            [5, 6, 8],
            [9, 6, 1],
            [6, 9, 8],
            [4, 7, 0],
            [4, 5, 8],
            [7, 4, 8],
            [12, 11, 8],
            [12, 9, 2],
            [9, 12, 8],
            [7, 10, 3],
            [10, 7, 8],
            [11, 10, 8],
        ],
    )

    assert_norm_equality(
        X.flatten(),
        [1.3905252276781415e+01, 3.0919890978059588e+00, 9.9320349979775846e-01],
        1.0e-12,
    )
    return
