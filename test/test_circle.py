# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_circle(h0=0.7, show=True):
    geo = dmsh.Circle([0.0, 0.0], 1.0)
    X, cells = dmsh.generate(geo, h0, show=show)

    # X = numpy.column_stack([X[0], X[1], numpy.zeros(X.shape[1])])
    # import meshio
    # meshio.write_points_cells("out.vtk", X, {"triangle": cells})

    assert numpy.array_equal(
        cells,
        [
            [3, 4, 6],
            [3, 5, 0],
            [5, 3, 6],
            [7, 4, 2],
            [4, 7, 6],
            [1, 3, 0],
            [4, 1, 2],
            [1, 4, 3],
        ],
    )

    tol = 1.0e-12
    X = X.flatten()

    ref_norm1 = 8.153528671130776
    assert abs(numpy.linalg.norm(X, ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 2.490883422065319
    assert abs(numpy.linalg.norm(X, ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 1.0
    assert abs(numpy.linalg.norm(X, ord=numpy.inf) - ref_norm_inf) < tol * ref_norm_inf
    return
