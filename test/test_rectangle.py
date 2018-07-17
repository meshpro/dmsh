# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_rectangle(h0=0.7, show=True):
    geo = dmsh.geometry.Rectangle(-1.0, +2.0, -1.0, +1.0)
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [2, 7, 6],
            [1, 2, 6],
            [1, 5, 0],
            [5, 1, 6],
            [3, 7, 2],
            [8, 3, 4],
            [3, 8, 7],
            [7, 11, 6],
            [11, 10, 6],
            [5, 10, 9],
            [10, 5, 6],
            [12, 11, 7],
            [12, 8, 13],
            [8, 12, 7],
        ],
    )

    tol = 1.0e-12
    ref_norm1 = 24.085394014345415
    assert abs(numpy.linalg.norm(X.flatten(), ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 5.465484083486126
    assert abs(numpy.linalg.norm(X.flatten(), ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 2.0
    assert (
        abs(numpy.linalg.norm(X.flatten(), ord=numpy.inf) - ref_norm_inf)
        < tol * ref_norm_inf
    )
    return
