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
            [8, 1, 3],
            [9, 7, 3],
            [8, 7, 0],
            [7, 8, 3],
            [11, 10, 4],
            [10, 9, 3],
            [1, 2, 3],
            [10, 6, 9],
            [6, 10, 11],
            [6, 11, 5],
        ],
    )

    assert_norm_equality(
        X.flatten(),
        [1.3905252276781415e+01, 3.0919890978059588e+00, 9.9320349979775846e-01],
        1.0e-12,
    )
    return


if __name__ == "__main__":
    test(h0=0.2)
