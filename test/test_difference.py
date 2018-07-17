# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_difference(h0=0.5, show=True):
    geo = dmsh.geometry.Difference(
        dmsh.geometry.Circle([-0.5, 0.0], 1.0), dmsh.geometry.Circle([+0.5, 0.0], 1.0)
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [9, 12, 11],
            [6, 8, 5],
            [9, 6, 7],
            [6, 4, 7],
            [4, 6, 0],
            [10, 9, 11],
            [10, 6, 9],
            [6, 10, 8],
            [3, 6, 5],
            [6, 3, 0],
            [4, 1, 2],
            [1, 4, 0],
        ],
    )
    tol = 1.0e-12

    ref_norm1 = 16.935078419974236
    assert abs(numpy.linalg.norm(X.flatten(), ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 3.986817558341568
    assert abs(numpy.linalg.norm(X.flatten(), ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 1.5
    assert (
        abs(numpy.linalg.norm(X.flatten(), ord=numpy.inf) - ref_norm_inf)
        < tol * ref_norm_inf
    )
    return


if __name__ == "__main__":
    test_difference(0.1, show=True)
