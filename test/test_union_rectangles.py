# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test_union(h0=0.9, show=True):
    geo = dmsh.Union(
        [dmsh.Rectangle(-1.0, +0.5, -1.0, +0.5), dmsh.Rectangle(-0.5, +1.0, -0.5, +1.0)]
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [11, 13, 4],
            [13, 7, 3],
            [7, 13, 11],
            [12, 11, 4],
            [8, 6, 2],
            [12, 6, 11],
            [6, 12, 5],
            [1, 7, 9],
            [10, 7, 11],
            [7, 10, 9],
            [6, 10, 11],
            [10, 6, 8],
            [9, 10, 0],
            [10, 8, 0],
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
