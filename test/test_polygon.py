# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test(h0=0.5, show=True):
    geo = dmsh.geometry.Polygon(
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
            [8, 5, 10],
            [0, 1, 3],
            [0, 4, 2],
            [4, 0, 3],
            [5, 4, 3],
            [4, 5, 8],
            [5, 9, 10],
            [6, 5, 3],
            [1, 6, 3],
            [9, 6, 7],
            [6, 9, 5],
        ],
    )

    tol = 1.0e-12
    ref_norm1 = 19.42381978164662
    assert abs(numpy.linalg.norm(X.flatten(), ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 5.020189840491164
    assert abs(numpy.linalg.norm(X.flatten(), ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 2.0
    assert (
        abs(numpy.linalg.norm(X.flatten(), ord=numpy.inf) - ref_norm_inf)
        < tol * ref_norm_inf
    )
    return


if __name__ == "__main__":
    test(h0=0.2)
