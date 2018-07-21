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

    print(cells)
    assert numpy.array_equal(
        cells,
        [
            [6, 9, 0],
            [8, 4, 5],
            [2, 4, 3],
            [2, 1, 5],
            [4, 2, 5],
            [9, 7, 10],
            [7, 9, 6],
            [7, 6, 3],
            [4, 7, 3],
            [8, 11, 4],
            [11, 7, 4],
            [7, 11, 10],
        ],
    )

    tol = 1.0e-12
    X = X.flatten()

    ref_norm1 = 11.918316817462927
    assert abs(numpy.linalg.norm(X, ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 2.8472772880595607
    assert abs(numpy.linalg.norm(X, ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 0.9664827275008776
    assert abs(numpy.linalg.norm(X, ord=numpy.inf) - ref_norm_inf) < tol * ref_norm_inf
    return


if __name__ == "__main__":
    test_halfspace(h0=0.1, show=True)
