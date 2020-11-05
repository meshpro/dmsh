import numpy
from helpers import assert_norm_equality, save

import dmsh


def test_boundary_step():
    """
    Finds the boundary step.

    Args:
    """
    geo = dmsh.Rectangle(-2.0, +2.0, -1.0, +1.0)

    # Check boundary steps
    out = geo.boundary_step([0.1, 0.0])
    assert numpy.all(numpy.abs(out - [2.0, 0.0]) < 1.0e-10)
    out = geo.boundary_step([0.0, 0.1])
    assert numpy.all(numpy.abs(out - [0.0, 1.0]) < 1.0e-10)
    out = geo.boundary_step([-0.1, 0.0])
    assert numpy.all(numpy.abs(out - [-2.0, 0.0]) < 1.0e-10)
    out = geo.boundary_step([0.0, -0.1])
    assert numpy.all(numpy.abs(out - [0.0, -1.0]) < 1.0e-10)

    out = geo.boundary_step([2.1, 0.037])
    assert numpy.all(numpy.abs(out - [2.0, 0.037]) < 1.0e-10)
    out = geo.boundary_step([0.037, 1.1])
    assert numpy.all(numpy.abs(out - [0.037, 1.0]) < 1.0e-10)
    out = geo.boundary_step([-2.1, 0.037])
    assert numpy.all(numpy.abs(out - [-2.0, 0.037]) < 1.0e-10)
    out = geo.boundary_step([0.037, -1.1])
    assert numpy.all(numpy.abs(out - [0.037, -1.0]) < 1.0e-10)

    out = geo.boundary_step([2.1, 1.1])
    assert numpy.all(numpy.abs(out - [2.0, 1.0]) < 1.0e-10)
    out = geo.boundary_step([-2.1, 1.1])
    assert numpy.all(numpy.abs(out - [-2.0, 1.0]) < 1.0e-10)
    out = geo.boundary_step([2.1, -1.1])
    assert numpy.all(numpy.abs(out - [2.0, -1.0]) < 1.0e-10)
    out = geo.boundary_step([-2.1, -1.1])
    assert numpy.all(numpy.abs(out - [-2.0, -1.0]) < 1.0e-10)


def test_rectangle(show=False):
    """
    Return the rectangle of the x y x and y coordinates.

    Args:
        show: (bool): write your description
    """
    geo = dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0)
    X, cells = dmsh.generate(geo, 0.1, show=show)

    ref_norms = [9.7543260517019439e02, 3.1710610220896264e01, 2.0]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


if __name__ == "__main__":
    X, cells = test_rectangle(show=False)
    save("rectangle.png", X, cells)
