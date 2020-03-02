import numpy

import dmsh
from helpers import assert_equality, assert_norm_equality, save


def test_boundary_step():
    geo = dmsh.Union([dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0)])
    geo.show()
    a = geo.boundary_step([-0.5, 0.9])
    assert numpy.array_equal(a, [-0.5, 1.0])

    a = geo.boundary_step([-0.5, 0.6])
    assert numpy.array_equal(a, [-0.5, 1.0])

    a = geo.boundary_step([0.05, 0.05])
    assert_equality(a, [-4.4469961425821203e-01, 9.9846976285554556e-01], 1.0e-10)


def test_union(show=False):
    geo = dmsh.Union([dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0)])
    X, cells = dmsh.generate(geo, 0.15, show=show, tol=1.0e-10)

    geo.plot()

    ref_norms = [3.0088043884612756e02, 1.5785099320497183e01, 1.5]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


if __name__ == "__main__":
    test_boundary_step()
    X, cells = test_union(show=False)
    save("union.png", X, cells)
