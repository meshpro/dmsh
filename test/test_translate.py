# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test(h0=0.9, show=True):
    geo = dmsh.Translation(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), [1.0, 1.0])
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [4, 1, 2],
            [1, 3, 0],
            [3, 1, 4],
            [5, 4, 2],
            [4, 5, 8],
            [7, 4, 8],
            [3, 7, 6],
            [7, 3, 4],
        ],
    )

    tol = 1.0e-12
    X = X.flatten()

    ref_norm1 = 22.5
    assert abs(numpy.linalg.norm(X, ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 6.9833474995775475
    assert abs(numpy.linalg.norm(X, ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 3.0
    assert abs(numpy.linalg.norm(X, ord=numpy.inf) - ref_norm_inf) < tol * ref_norm_inf
    return


if __name__ == "__main__":
    test(h0=0.1)
