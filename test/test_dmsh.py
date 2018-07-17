# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_circle(h0=0.7, show=True):
    geo = dmsh.geometry.Circle([0.0, 0.0], 1.0)
    X, cells = dmsh.generate(geo, h0, show=show)

    # X = numpy.column_stack([X[0], X[1], numpy.zeros(X.shape[1])])
    # import meshio
    # meshio.write_points_cells("out.vtk", X, {"triangle": cells})

    X = X.T

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

    ref_norm1 = 8.153528671130776
    assert abs(numpy.linalg.norm(X.flatten(), ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 2.490883422065319
    assert abs(numpy.linalg.norm(X.flatten(), ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 1.0
    assert (
        abs(numpy.linalg.norm(X.flatten(), ord=numpy.inf) - ref_norm_inf)
        < tol * ref_norm_inf
    )
    return


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
    X = X.T
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


def test_rectangle(h0=0.7, show=True):
    geo = dmsh.geometry.Rectangle(-1.0, +2.0, -1.0, +1.0)
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [2, 7, 6],
            [1, 2, 6],
            [1, 5, 0],
            [5, 1, 6],
            [3, 7, 2],
            [8, 3, 4],
            [3, 8, 7],
            [7, 11, 6],
            [11, 10, 6],
            [5, 10, 9],
            [10, 5, 6],
            [12, 11, 7],
            [12, 8, 13],
            [8, 12, 7],
        ],
    )

    tol = 1.0e-12
    X = X.T
    ref_norm1 = 24.085394014345415
    assert abs(numpy.linalg.norm(X.flatten(), ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 5.465484083486126
    assert abs(numpy.linalg.norm(X.flatten(), ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 2.0
    assert (
        abs(numpy.linalg.norm(X.flatten(), ord=numpy.inf) - ref_norm_inf)
        < tol * ref_norm_inf
    )
    return


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

    X = X.T
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


def test_intersection(h0=0.5, show=True):
    geo = dmsh.geometry.Intersection(
        [dmsh.geometry.Circle([-0.5, 0.0], 1.0), dmsh.geometry.Circle([+0.5, 0.0], 1.0)]
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    print(cells)
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

    X = X.T
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


if __name__ == "__main__":
    test_intersection(0.1, show=False)
