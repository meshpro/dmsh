import numpy as np
from helpers import assert_norm_equality, save

import dmsh


def test_boundary_step():
    geo = dmsh.Rectangle(-2.0, +2.0, -1.0, +1.0)

    # Check boundary steps
    out = geo.boundary_step([0.1, 0.0])
    assert np.all(np.abs(out - [2.0, 0.0]) < 1.0e-10)
    out = geo.boundary_step([0.0, 0.1])
    assert np.all(np.abs(out - [0.0, 1.0]) < 1.0e-10)
    out = geo.boundary_step([-0.1, 0.0])
    assert np.all(np.abs(out - [-2.0, 0.0]) < 1.0e-10)
    out = geo.boundary_step([0.0, -0.1])
    assert np.all(np.abs(out - [0.0, -1.0]) < 1.0e-10)

    out = geo.boundary_step([2.1, 0.037])
    assert np.all(np.abs(out - [2.0, 0.037]) < 1.0e-10)
    out = geo.boundary_step([0.037, 1.1])
    assert np.all(np.abs(out - [0.037, 1.0]) < 1.0e-10)
    out = geo.boundary_step([-2.1, 0.037])
    assert np.all(np.abs(out - [-2.0, 0.037]) < 1.0e-10)
    out = geo.boundary_step([0.037, -1.1])
    assert np.all(np.abs(out - [0.037, -1.0]) < 1.0e-10)

    out = geo.boundary_step([2.1, 1.1])
    assert np.all(np.abs(out - [2.0, 1.0]) < 1.0e-10)
    out = geo.boundary_step([-2.1, 1.1])
    assert np.all(np.abs(out - [-2.0, 1.0]) < 1.0e-10)
    out = geo.boundary_step([2.1, -1.1])
    assert np.all(np.abs(out - [2.0, -1.0]) < 1.0e-10)
    out = geo.boundary_step([-2.1, -1.1])
    assert np.all(np.abs(out - [-2.0, -1.0]) < 1.0e-10)


def test_rectangle(show=False):
    geo = dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0)
    X, cells = dmsh.generate(geo, 0.1, show=show)

    ref_norms = [9.7542855197092831e02, 3.1710489987948261e01, 2.0000000000000000e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


def test_duplicate_points(show=False):
    # https://github.com/nschloe/dmsh/issues/66
    geo = dmsh.Rectangle(0.0, 1.8, 0.0, 0.41)

    points, triangles = dmsh.generate(geo, 0.2, tol=1e-5, show=show)

    tmp = np.ascontiguousarray(points)
    assert points.shape[0] == np.unique(tmp.view([("", tmp.dtype)] * tmp.shape[1])).shape[0]


if __name__ == "__main__":
    test_duplicate_points(show=False)
    # X, cells = test_rectangle(show=False)
    # save("rectangle.png", X, cells)
