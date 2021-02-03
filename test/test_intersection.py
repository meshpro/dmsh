import numpy as np
from helpers import assert_norm_equality, save

import dmsh


def test_intersection(show=False):
    geo = dmsh.Intersection(
        [dmsh.Circle([0.0, -0.5], 1.0), dmsh.Circle([0.0, +0.5], 1.0)]
    )
    X, cells = dmsh.generate(geo, 0.1, show=show, tol=1.0e-10)

    geo.plot()

    ref_norms = [8.6619344515576643e01, 6.1599895121114256e00, 8.6602540378466342e-01]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


def test_intersection_circles(show=False):
    angles = np.pi * np.array([3.0 / 6.0, 7.0 / 6.0, 11.0 / 6.0])
    geo = dmsh.Intersection(
        [
            dmsh.Circle([np.cos(angles[0]), np.sin(angles[0])], 1.5),
            dmsh.Circle([np.cos(angles[1]), np.sin(angles[1])], 1.5),
            dmsh.Circle([np.cos(angles[2]), np.sin(angles[2])], 1.5),
        ]
    )
    X, cells = dmsh.generate(geo, 0.1, show=show, tol=1.0e-10)

    ref_norms = [6.7650924353448104e01, 5.0559320826807497e00, 7.2474487138537913e-01]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


def test_boundary_step():
    geo = dmsh.Intersection(
        [dmsh.Circle([0.0, -0.5], 1.0), dmsh.Circle([0.0, +0.5], 1.0)]
    )
    pts = np.array([[0.0, -5.0], [0.0, 4.1]])
    pts = geo.boundary_step(pts.T).T
    ref = np.array([[0.0, -0.5], [0.0, 0.5]])
    assert np.all(np.abs(pts - ref) < 1.0e-10)

    pts = np.array([[0.0, -0.1], [0.0, 0.1]])
    pts = geo.boundary_step(pts.T).T
    ref = np.array([[0.0, -0.5], [0.0, 0.5]])
    assert np.all(np.abs(pts - ref) < 1.0e-10)


def test_boundary_step2():
    geo = dmsh.Intersection(
        [dmsh.Circle([0.0, -0.5], 1.0), dmsh.Circle([0.0, +0.5], 1.0)]
    )
    np.random.seed(0)
    pts = np.random.uniform(-1.0, 1.0, (2, 100))
    pts = geo.boundary_step(pts)
    # geo.plot()
    # import matplotlib.pyplot as plt
    # plt.plot(pts[0], pts[1], "xk")
    # plt.show()
    assert np.all(np.abs(geo.dist(pts)) < 1.0e-7)


if __name__ == "__main__":
    X, cells = test_intersection(show=True)
    save("intersection.png", X, cells)
    # test_boundary_step2()
