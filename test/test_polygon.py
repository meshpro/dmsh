import numpy
from helpers import assert_norm_equality

import dmsh


def test(show=False):
    geo = dmsh.Polygon(
        [
            [0.0, 0.0],
            [1.1, 0.0],
            [1.2, 0.5],
            [0.7, 0.6],
            [2.0, 1.0],
            [1.0, 2.0],
            [0.5, 1.5],
        ]
    )
    # geo.show()
    X, cells = dmsh.generate(geo, 0.1, show=show)

    ref_norms = [4.1454432512302594e02, 2.1854133564894923e01, 2.0000000000000000e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-5)
    return X, cells


def test_boundary_step2():
    geo = dmsh.Polygon(
        [
            [0.0, 0.0],
            [1.1, 0.0],
            [1.2, 0.5],
            [0.7, 0.6],
            [2.0, 1.0],
            [1.0, 2.0],
            [0.5, 1.5],
        ]
    )
    numpy.random.seed(0)
    pts = numpy.random.uniform(-2.0, 2.0, (2, 100))
    pts = geo.boundary_step(pts)
    # geo.plot()
    # import matplotlib.pyplot as plt
    # plt.plot(pts[0], pts[1], "xk")
    # plt.show()
    # print(geo.dist(pts).shape)
    dist = geo.dist(pts)
    print(numpy.max(numpy.abs(dist)))
    assert numpy.all(numpy.abs(dist) < 1.0e-12)


if __name__ == "__main__":
    # from helpers import save
    # X, cells = test(show=False)
    # save("polygon.svg", X, cells)
    test_boundary_step2()
