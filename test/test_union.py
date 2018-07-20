# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_union(h0=0.9, show=True):
    geo = dmsh.geometry.Union(
        [dmsh.geometry.Circle([-0.5, 0.0], 1.0), dmsh.geometry.Circle([+0.5, 0.0], 1.0)]
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [10, 6, 7],
            [6, 4, 7],
            [1, 6, 10],
            [0, 4, 6],
            [3, 0, 6],
            [2, 3, 5],
            [3, 2, 0],
            [1, 9, 6],
            [9, 3, 6],
            [8, 9, 1],
            [9, 8, 5],
            [3, 9, 5],
        ],
    )
    tol = 1.0e-12

    ref_norm1 = 14.35844293007027
    assert abs(numpy.linalg.norm(X.flatten(), ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 3.7470363552767827
    assert abs(numpy.linalg.norm(X.flatten(), ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 1.5
    assert (
        abs(numpy.linalg.norm(X.flatten(), ord=numpy.inf) - ref_norm_inf)
        < tol * ref_norm_inf
    )
    return


if __name__ == "__main__":
    test_union(h0=0.3)
