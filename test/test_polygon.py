# -*- coding: utf-8 -*-
#
import numpy

import dmsh


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

    tol = 1.0e-12
    X = X.flatten()

    ref_norm1 = 19.932424285039232
    assert abs(numpy.linalg.norm(X, ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 5.007357002540644
    assert abs(numpy.linalg.norm(X, ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 2.0
    assert abs(numpy.linalg.norm(X, ord=numpy.inf) - ref_norm_inf) < tol * ref_norm_inf
    return


if __name__ == "__main__":
    test(h0=0.2)
