import numpy
from helpers import assert_equality, assert_norm_equality

import dmsh


def test_union_circles(show=False):
    geo = dmsh.Union([dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0)])
    X, cells = dmsh.generate(geo, 0.15, show=show, tol=1.0e-5)

    geo.plot()

    ref_norms = [3.0087320808275558e+02, 1.5784709390397147e+01, 1.5000000000000000e+00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


def test_union_rectangles(show=False):
    geo = dmsh.Union(
        [dmsh.Rectangle(-1.0, +0.5, -1.0, +0.5), dmsh.Rectangle(-0.5, +1.0, -0.5, +1.0)]
    )
    X, cells = dmsh.generate(geo, 0.15, show=show, tol=1.0e-5)

    ref_norms = [1.8420561245539048e+02, 1.1269215671323987e+01, 1.0000000000000000e+00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


def test_union_three_circles(show=False):
    angles = numpy.pi * numpy.array([3.0 / 6.0, 7.0 / 6.0, 11.0 / 6.0])
    geo = dmsh.Union(
        [
            dmsh.Circle([numpy.cos(angles[0]), numpy.sin(angles[0])], 1.0),
            dmsh.Circle([numpy.cos(angles[1]), numpy.sin(angles[1])], 1.0),
            dmsh.Circle([numpy.cos(angles[2]), numpy.sin(angles[2])], 1.0),
        ]
    )
    X, cells = dmsh.generate(geo, 0.2, show=show, tol=1.0e-5)

    ref_norms = [4.0370458355600903e02, 2.1155724692606825e01, 2.0000000000000000e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


def test_boundary_step():
    geo = dmsh.Union([dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0)])
    a = geo.boundary_step([-0.5, 0.9])
    assert numpy.array_equal(a, [-0.5, 1.0])

    a = geo.boundary_step([-0.5, 0.6])
    assert numpy.array_equal(a, [-0.5, 1.0])

    a = geo.boundary_step([0.05, 0.05])
    assert_equality(a, [-4.4469961425821203e-01, 9.9846976285554556e-01], 1.0e-10)

    pts = numpy.array([[-5.0, 0.0], [4.1, 0.0]])
    pts = geo.boundary_step(pts.T).T
    ref = numpy.array([[-1.5, 0.0], [1.5, 0.0]])
    assert numpy.all(numpy.abs(pts - ref) < 1.0e-10)

    pts = numpy.array([[-0.9, 0.0], [1.1, 0.0]])
    pts = geo.boundary_step(pts.T).T
    ref = numpy.array([[-1.5, 0.0], [1.5, 0.0]])
    assert numpy.all(numpy.abs(pts - ref) < 1.0e-10)


def test_boundary_step2():
    geo = dmsh.Union([dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0)])
    numpy.random.seed(0)
    pts = numpy.random.uniform(-2.0, 2.0, (2, 100))
    pts = geo.boundary_step(pts)
    # geo.plot()
    # import matplotlib.pyplot as plt
    # plt.plot(pts[0], pts[1], "xk")
    # plt.show()
    assert numpy.all(numpy.abs(geo.dist(pts)) < 1.0e-12)


if __name__ == "__main__":
    # from helpers import save
    X, cells = test_union_circles(show=True)
    # save("union.png", X, cells)
    # test_boundary_step2()
