# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_union(h0=1.4, show=True):
    angles = numpy.pi * numpy.array([3.0 / 6.0, 7.0 / 6.0, 11.0 / 6.0])
    geo = dmsh.Union(
        [
            dmsh.Circle([numpy.cos(angles[0]), numpy.sin(angles[0])], 1.0),
            dmsh.Circle([numpy.cos(angles[1]), numpy.sin(angles[1])], 1.0),
            dmsh.Circle([numpy.cos(angles[2]), numpy.sin(angles[2])], 1.0),
        ]
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [0, 9, 7],
            [9, 3, 8],
            [3, 2, 8],
            [3, 0, 2],
            [0, 3, 9],
            [0, 1, 2],
            [1, 0, 7],
            [4, 1, 7],
            [1, 5, 2],
            [2, 5, 8],
            [5, 6, 8],
        ],
    )

    tol = 1.0e-12
    X = X.flatten()

    ref_norm1 = 17.387135311724656
    assert abs(numpy.linalg.norm(X, ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 4.9055252285569075
    assert abs(numpy.linalg.norm(X, ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 2.0
    assert abs(numpy.linalg.norm(X, ord=numpy.inf) - ref_norm_inf) < tol * ref_norm_inf
    return


if __name__ == "__main__":
    test_union(h0=0.3)
