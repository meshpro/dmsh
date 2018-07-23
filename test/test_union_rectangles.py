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

    print(cells)
    assert numpy.array_equal(
        cells,
        [
            [12, 7, 3],
            [8, 7, 11],
            [7, 8, 9],
            [9, 8, 0],
            [8, 10, 0],
            [10, 6, 2],
            [6, 8, 11],
            [8, 6, 10],
            [7, 14, 11],
            [14, 7, 12],
            [14, 6, 11],
            [6, 14, 13],
            [14, 12, 4],
            [13, 14, 4],
            [1, 7, 9],
            [5, 6, 13],
        ],
    ), cells

    assert_norm_equality(
        X.flatten(),
        [1.8517346917955408e+01, 3.8567381587761487e+00, 1.0],
        1.0e-12,
    )
    return


if __name__ == "__main__":
    test_union(h0=0.3)
