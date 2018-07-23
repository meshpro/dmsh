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
            [2, 5, 4],
            [5, 2, 0],
            [8, 5, 6],
            [5, 8, 1],
            [5, 3, 6],
            [3, 5, 0],
            [5, 7, 4],
            [7, 5, 1],
        ],
    ), cells

    assert_norm_equality(
        X.flatten(), [1.2209924970305869e+01, 3.6101367670000442e+00, 1.5], 1.0e-12
    )
    return


if __name__ == "__main__":
    test_union(h0=0.3)
