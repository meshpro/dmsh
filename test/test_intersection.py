# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_intersection(h0=0.5, show=True):
    geo = dmsh.geometry.Intersection(
        [dmsh.geometry.Circle([-0.5, 0.0], 1.0), dmsh.geometry.Circle([+0.5, 0.0], 1.0)]
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [4, 6, 3],
            [7, 4, 5],
            [6, 7, 8],
            [7, 6, 4],
            [1, 4, 3],
            [4, 2, 5],
            [2, 1, 0],
            [1, 2, 4],
        ],
    )
    tol = 1.0e-12

    ref_norm1 = 6.2249301192580635
    assert abs(numpy.linalg.norm(X.flatten(), ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 1.878680087718239
    assert abs(numpy.linalg.norm(X.flatten(), ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 0.8578313012434592
    assert (
        abs(numpy.linalg.norm(X.flatten(), ord=numpy.inf) - ref_norm_inf)
        < tol * ref_norm_inf
    )
    return
