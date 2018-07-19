# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_rectangle(h0=0.8, show=True):
    geo = dmsh.geometry.Rectangle(-1.0, +2.0, -1.0, +1.0)
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

    tol = 1.0e-12
    ref_norm1 = 23.50010568219156
    assert abs(numpy.linalg.norm(X.flatten(), ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 5.38518180031268
    assert abs(numpy.linalg.norm(X.flatten(), ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 2.0
    assert (
        abs(numpy.linalg.norm(X.flatten(), ord=numpy.inf) - ref_norm_inf)
        < tol * ref_norm_inf
    )
    return
