# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_ellipse(h0=0.7, show=True):
    geo = dmsh.geometry.Ellipse([0.0, 0.0], 2.0, 1.0)
    X, cells = dmsh.generate(geo, h0, show=show)

    # X = numpy.column_stack([X[0], X[1], numpy.zeros(X.shape[1])])
    # import meshio
    # meshio.write_points_cells("out.vtk", X, {"triangle": cells})

    assert numpy.array_equal(
        cells,
        [
            [14, 8, 9],
            [8, 4, 9],
            [6, 10, 5],
            [0, 6, 5],
            [14, 13, 8],
            [3, 4, 8],
            [6, 11, 10],
            [1, 6, 0],
            [3, 7, 2],
            [7, 3, 8],
            [7, 1, 2],
            [1, 7, 6],
            [7, 13, 12],
            [13, 7, 8],
            [11, 7, 12],
            [7, 11, 6],
        ],
    )

    tol = 1.0e-12
    ref_norm1 = 22.435407673283038
    assert abs(numpy.linalg.norm(X.flatten(), ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 5.121569408475023
    assert abs(numpy.linalg.norm(X.flatten(), ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 2.0
    assert (
        abs(numpy.linalg.norm(X.flatten(), ord=numpy.inf) - ref_norm_inf)
        < tol * ref_norm_inf
    )
    return
