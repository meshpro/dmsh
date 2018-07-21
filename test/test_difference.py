# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_difference(h0=0.5, show=True):
    geo = dmsh.geometry.Difference(
        dmsh.geometry.Circle([-0.5, 0.0], 1.0), dmsh.geometry.Circle([+0.5, 0.0], 1.0)
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    print(cells)
    assert numpy.array_equal(
        cells,
        [
            [11, 8, 9],
            [12, 8, 11],
            [8, 6, 9],
            [6, 8, 2],
            [11, 1, 14],
            [13, 11, 14],
            [13, 12, 11],
            [5, 8, 7],
            [8, 5, 2],
            [8, 10, 7],
            [10, 8, 12],
            [3, 6, 2],
            [0, 3, 4],
            [3, 0, 6],
        ],
    )

    tol = 1.0e-12
    X = X.flatten()

    ref_norm1 = 19.873204312684507
    assert abs(numpy.linalg.norm(X, ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 4.316711705494604
    assert abs(numpy.linalg.norm(X, ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 1.4834095117965043
    assert abs(numpy.linalg.norm(X, ord=numpy.inf) - ref_norm_inf) < tol * ref_norm_inf
    return


if __name__ == "__main__":
    test_difference(0.1, show=True)
