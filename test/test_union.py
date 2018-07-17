# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_union(h0=0.7, show=True):
    geo = dmsh.geometry.Union(
        [dmsh.geometry.Circle([-0.5, 0.0], 1.0), dmsh.geometry.Circle([+0.5, 0.0], 1.0)]
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    # X = numpy.column_stack([X[0], X[1], numpy.zeros(X.shape[1])])
    # import meshio
    # meshio.write_points_cells("out.vtk", X, {"triangle": cells})

    assert numpy.array_equal(
        cells,
        [
            [4, 7, 3],
            [7, 4, 8],
            [0, 4, 3],
            [4, 0, 1],
            [5, 4, 1],
            [4, 5, 8],
            [9, 5, 6],
            [5, 9, 8],
            [5, 2, 6],
            [2, 5, 1],
        ],
    )
    tol = 1.0e-12

    ref_norm1 = 13.500627895090055
    assert abs(numpy.linalg.norm(X.flatten(), ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 3.701888886208963
    assert abs(numpy.linalg.norm(X.flatten(), ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 1.5
    assert (
        abs(numpy.linalg.norm(X.flatten(), ord=numpy.inf) - ref_norm_inf)
        < tol * ref_norm_inf
    )
    return
