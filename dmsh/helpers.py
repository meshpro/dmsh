# -*- coding: utf-8 -*-
#
import matplotlib.pyplot as plt
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


def find_feature_points(geo0, geo1, num_steps=10):
    t0, t1 = numpy.meshgrid(numpy.linspace(0.0, 1.0, 11), numpy.linspace(0.0, 1.0, 11))
    t = numpy.array([t0, t1]).reshape(2, -1)
    # t = numpy.random.rand(2, 100)

    plt.plot(t[0], t[1], ".")
    plt.axis("square")
    plt.show()

    tol = 1.0e-15

    # multi_newton(x0, is_inside, boundary_step, tol, max_num_steps=10):
    solutions = []
    for k in range(num_steps):
        f_t = geo0.parametrization(t[0]) - geo1.parametrization(t[1])

        f_dot_f = numpy.einsum("ij,ij->j", f_t, f_t)
        is_sol = f_dot_f < tol

        if numpy.any(is_sol):
            solutions.append(t[:, is_sol])
            # remove all converged solutions
            t = t[:, ~is_sol]
            f_t = f_t[:, ~is_sol]

        jac_t = numpy.stack([geo0.dp_dt(t[0]), -geo1.dp_dt(t[1])])

        # Kick out singular matrices
        det = jac_t[0, 0] * jac_t[1, 1] - jac_t[0, 1] * jac_t[1, 0]
        is_singular = numpy.abs(det) < 1.0e-13
        if numpy.any(is_singular):
            t = t[:, ~is_singular]
            f_t = f_t[:, ~is_singular]
            jac_t = jac_t[..., ~is_singular]

        sols = []
        for k in range(f_t.shape[-1]):
            sols.append(numpy.linalg.solve(jac_t[..., k].T, f_t[:, k]))
        sols = numpy.array(sols).T

        t -= sols

        # Kick out everything that leaves the unit square
        still_good = numpy.all((0.0 <= t) & (t <= 1.0), axis=0)
        t = t[:, still_good]

        # plt.plot(t[0], t[1], '.')
        # plt.axis("square")
        # plt.show()

    solutions = numpy.column_stack(solutions)
    unique_sols = unique_float_cols(solutions)

    points = geo0.parametrization(unique_sols[0])
    return points.T


def unique_float_cols(data, tol=1.0e-10):
    """In a (k, n) array `data`, find the unique columns.
    """
    # Sort the columns
    data = numpy.sort(data, axis=-1)

    # Find where two columns differ normwise by more than tol
    diff = data[:, 1:] - data[:, :-1]
    norm_diff = numpy.einsum("ij,ij->j", diff, diff)
    cut = norm_diff > tol

    # Take the first column and all where there is a cut
    return numpy.column_stack([data[:, 0], data[:, 1:][:, cut]])
