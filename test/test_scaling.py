# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test(h0=1.8, show=True):
    geo = dmsh.geometry.Scaling(dmsh.geometry.Rectangle(-1.0, +2.0, -1.0, +1.0), 2.0)
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [4, 1, 2],
            [1, 3, 0],
            [3, 1, 4],
            [5, 4, 2],
            [5, 8, 4],
            [8, 7, 4],
            [3, 7, 6],
            [7, 3, 4],
        ],
    )

    tol = 1.0e-12
    ref_norm1 = 33.0
    assert abs(numpy.linalg.norm(X.flatten(), ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 9.331044917825025
    assert abs(numpy.linalg.norm(X.flatten(), ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 4.0
    assert (
        abs(numpy.linalg.norm(X.flatten(), ord=numpy.inf) - ref_norm_inf)
        < tol * ref_norm_inf
    )
    return


if __name__ == "__main__":
    test(h0=0.1)
