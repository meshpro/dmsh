# -*- coding: utf-8 -*-
#
import numpy


def unique_rows(a):
    # The cleaner alternative `numpy.unique(a, axis=0)` is slow; cf.
    # <https://github.com/numpy/numpy/issues/11136>.
    b = numpy.ascontiguousarray(a).view(
        numpy.dtype((numpy.void, a.dtype.itemsize * a.shape[1]))
    )
    a_unique, inv, cts = numpy.unique(b, return_inverse=True, return_counts=True)
    a_unique = a_unique.view(a.dtype).reshape(-1, a.shape[1])
    return a_unique, inv, cts


def multi_newton(x0, is_inside, boundary_step, tol, max_num_steps=10):
    """Newton's minimization method for multiple starting points.
    """
    x = x0.copy()
    fx = is_inside(x.T)

    k = 0
    mask = numpy.abs(fx) > tol
    while numpy.any(mask):
        x[mask] = boundary_step(x[mask].T).T
        fx = is_inside(x.T)
        mask = numpy.abs(fx) > tol
        k += 1
        if k >= max_num_steps:
            break

    return x


def show(pts, cells, geo):
    import matplotlib.pyplot as plt

    eps = 1.0e-10
    is_inside = geo.isinside(pts.T) < eps
    plt.plot(pts[is_inside, 0], pts[is_inside, 1], ".")
    plt.plot(pts[~is_inside, 0], pts[~is_inside, 1], ".", color="r")
    plt.triplot(pts[:, 0], pts[:, 1], cells)
    plt.axis("square")

    try:
        geo.plot()
    except AttributeError:
        pass

    plt.show()
    return
