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


def multi_newton(x0, geo, tol, max_num_steps=10):
    """Newton's minimization method for multiple starting points.
    """
    x = x0.copy()
    fx = geo.isinside(x)

    k = 0
    while numpy.any(numpy.abs(fx) > tol):
        jac = geo.jac2(x)
        hess = geo.hessian2(x)
        p = -numpy.linalg.solve(numpy.moveaxis(hess, -1, 0), jac.T).T
        x += p
        fx = geo.isinside(x)
        k += 1
        if k >= max_num_steps:
            break

    return x


def show(pts, cells, geo):
    import matplotlib.pyplot as plt
    eps = 1.0e-10
    is_inside = geo.isinside(pts) < eps
    plt.plot(pts[0, is_inside], pts[1, is_inside], ".")
    plt.plot(pts[0, ~is_inside], pts[1, ~is_inside], ".", color="r")
    plt.triplot(pts[0], pts[1], cells)
    plt.axis("square")
    plt.show()
    return
