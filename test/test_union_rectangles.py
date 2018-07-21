# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_union(h0=0.9, show=True):
    geo = dmsh.Union(
        [
            dmsh.Rectangle(-1.0, +0.5, -1.0, +0.5),
            dmsh.Rectangle(-0.5, +1.0, -0.5, +1.0),
        ]
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

    tol = 1.0e-12
    X = X.flatten()

    ref_norm1 = 18.117948759942017
    assert abs(numpy.linalg.norm(X, ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 3.8159074945459786
    assert abs(numpy.linalg.norm(X, ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 1.0
    assert abs(numpy.linalg.norm(X, ord=numpy.inf) - ref_norm_inf) < tol * ref_norm_inf
    return


if __name__ == "__main__":
    test_union(h0=0.3)
