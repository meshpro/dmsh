# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_halfspace(h0=0.4, show=True):
    geo = dmsh.geometry.Intersection(
        [
            dmsh.geometry.HalfSpace(numpy.sqrt(0.5) * numpy.array([1.0, 1.0]), 0.0),
            dmsh.geometry.Circle([0.0, 0.0], 1.0),
        ]
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [2, 0, 3],
            [2, 5, 1],
            [5, 2, 3],
            [5, 4, 1],
            [4, 8, 7],
            [8, 5, 9],
            [5, 8, 4],
            [6, 5, 3],
            [5, 6, 9],
        ],
    )
    tol = 1.0e-12

    ref_norm1 = 9.885121075672002
    assert abs(numpy.linalg.norm(X.flatten(), ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 2.596579298582527
    assert abs(numpy.linalg.norm(X.flatten(), ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 0.989212245376449
    assert (
        abs(numpy.linalg.norm(X.flatten(), ord=numpy.inf) - ref_norm_inf)
        < tol * ref_norm_inf
    )
    return


if __name__ == "__main__":
    test_halfspace(h0=0.1, show=False)
