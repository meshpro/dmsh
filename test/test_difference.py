# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test_difference(h0=0.5, show=True):
    geo = dmsh.Difference(dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0))
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [2, 4, 6],
            [0, 5, 3],
            [4, 5, 7],
            [5, 2, 3],
            [5, 4, 2],
            [11, 9, 1],
            [9, 11, 10],
            [9, 8, 7],
            [8, 9, 10],
            [8, 4, 7],
            [8, 10, 6],
            [4, 8, 6],
        ],
    )

    assert_norm_equality(
        X.flatten(), [1.4961168342345637e+01, 3.6391828386432765e+00, 1.5], 1.0e-12
    )
    return


if __name__ == "__main__":
    test_difference(0.1, show=True)
