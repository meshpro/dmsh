import numpy
from helpers import assert_norm_equality, save

import dmsh


def test_intersection(show=False):
    geo = dmsh.Intersection(
        [dmsh.Circle([0.0, -0.5], 1.0), dmsh.Circle([0.0, +0.5], 1.0)]
    )
    X, cells = dmsh.generate(geo, 0.1, show=show, tol=1.0e-10)

    geo.plot()

    ref_norms = [8.6619344595913475e01, 6.1599895121114274e00, 8.6602540378466342e-01]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


def test_boundary_step():
    geo = dmsh.Intersection(
        [dmsh.Circle([0.0, -0.5], 1.0), dmsh.Circle([0.0, +0.5], 1.0)]
    )
    pts = numpy.array([[0.0, -5.0], [0.0, 4.1]])
    pts = geo.boundary_step(pts.T).T
    ref = numpy.array([[0.0, -0.5], [0.0, 0.5]])
    assert numpy.all(numpy.abs(pts - ref) < 1.0e-10)

    pts = numpy.array([[0.0, -0.1], [0.0, 0.1]])
    pts = geo.boundary_step(pts.T).T
    ref = numpy.array([[0.0, -0.5], [0.0, 0.5]])
    assert numpy.all(numpy.abs(pts - ref) < 1.0e-10)


def test_boundary_step2():
    geo = dmsh.Intersection(
        [dmsh.Circle([0.0, -0.5], 1.0), dmsh.Circle([0.0, +0.5], 1.0)]
    )
    numpy.random.seed(0)
    pts = numpy.random.uniform(-1.0, 1.0, (2, 100))
    pts = geo.boundary_step(pts)
    geo.plot()
    import matplotlib.pyplot as plt

    plt.plot(pts[0], pts[1], "xk")
    plt.show()
    tol = 1.0e-7
    assert numpy.all(numpy.abs(geo.dist(pts)) < tol)


if __name__ == "__main__":
    X, cells = test_intersection(show=True)
    save("intersection.png", X, cells)
    # test_boundary_step2()
