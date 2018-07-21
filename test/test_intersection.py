# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_intersection(h0=0.5, show=True):
    geo = dmsh.Intersection(
        [dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0)]
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [6, 8, 5],
            [9, 6, 7],
            [3, 6, 5],
            [6, 4, 7],
            [10, 8, 6],
            [9, 10, 6],
            [10, 9, 1],
            [3, 2, 6],
            [2, 4, 6],
            [2, 3, 0],
        ],
    )

    tol = 1.0e-12
    X = X.flatten()

    ref_norm1 = 8.08910757330533
    assert abs(numpy.linalg.norm(X, ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 2.088785479836415
    assert abs(numpy.linalg.norm(X, ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 0.8660254037740615
    assert abs(numpy.linalg.norm(X, ord=numpy.inf) - ref_norm_inf) < tol * ref_norm_inf
    return


if __name__ == "__main__":
    test_intersection(0.1)
