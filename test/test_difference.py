import numpy
from helpers import assert_norm_equality

import dmsh


def test_difference(show=False):
    geo = dmsh.Difference(dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0))
    X, cells = dmsh.generate(geo, 0.1, show=show)

    geo.plot()

    ref_norms = [2.9445582949135735e02, 1.5856370081862632e01, 1.4999999025187867e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-6)
    return X, cells


def test_boundary_step():
    geo = dmsh.Difference(dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0))
    numpy.random.seed(0)
    pts = numpy.random.uniform(-1.0, 1.0, (2, 100))
    pts = geo.boundary_step(pts)
    # geo.plot()
    # import matplotlib.pyplot as plt
    # plt.plot(pts[0], pts[1], "xk")
    # plt.show()
    tol = 1.0e-7
    assert numpy.all(numpy.abs(geo.dist(pts)) < tol)


if __name__ == "__main__":
    # from helpers import save
    # X, cells = test_difference(show=True)
    # save("difference.png", X, cells)
    test_boundary_step()
