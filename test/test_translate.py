# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test(h0=0.9, show=True):
    geo = dmsh.Translation(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), [1.0, 1.0])
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [4, 3, 0],
            [3, 4, 6],
            [1, 4, 0],
            [5, 1, 2],
            [1, 5, 4],
            [4, 7, 6],
            [7, 5, 8],
            [5, 7, 4],
        ],
    ), cells

    assert_norm_equality(X.flatten(), [2.25e+01, 6.9833473916505637e+00, 3.0], 1.0e-12)
    return


if __name__ == "__main__":
    test(h0=0.1)
