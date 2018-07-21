# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test(h0=0.9, show=True):
    geo = dmsh.Stretch(
        dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), [1.0, 1.0]
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [8, 4, 5],
            [1, 4, 0],
            [4, 1, 5],
            [7, 6, 3],
            [9, 8, 5],
            [6, 9, 5],
            [6, 2, 3],
            [1, 2, 5],
            [2, 6, 5],
            [10, 9, 6],
            [10, 7, 11],
            [7, 10, 6],
        ],
    )

    tol = 1.0e-12
    X = X.flatten()
    ref_norm1 = 25.123192052335945
    assert abs(numpy.linalg.norm(X, ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 6.217716013864884
    assert abs(numpy.linalg.norm(X, ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 2.621320343559643
    assert abs(numpy.linalg.norm(X, ord=numpy.inf) - ref_norm_inf) < tol * ref_norm_inf
    return


if __name__ == "__main__":
    test(h0=0.1)
