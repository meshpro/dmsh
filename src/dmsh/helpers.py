from __future__ import annotations

from typing import Callable

import numpy as np


def multi_newton(
    x0: np.ndarray,
    is_inside: Callable,
    boundary_step: Callable,
    tol: float,
    max_num_steps: int = 10,
) -> np.ndarray:
    """Newton's minimization method for multiple starting points."""
    x = x0.copy()
    fx = is_inside(x.T)

    k = 0
    mask = np.abs(fx) > tol
    while np.any(mask):
        x[mask] = boundary_step(x[mask].T).T
        fx = is_inside(x.T)
        mask = np.abs(fx) > tol
        k += 1
        if k >= max_num_steps:
            break

    return x


def show(pts, cells, geo, title: str | None = None, full_screen: bool = True):
    import matplotlib.pyplot as plt

    eps = 1.0e-10
    # highlight outside points in C3, and points which aren't part of any cell in C4
    is_part_of_cell = np.zeros(len(pts), dtype=bool)
    is_part_of_cell[cells.flat] = True

    is_inside = geo.dist(pts.T) < eps

    sp = pts[is_inside & is_part_of_cell]
    plt.plot(sp[:, 0], sp[:, 1], ".", color="C0")
    sp = pts[~is_inside]
    plt.plot(sp[:, 0], sp[:, 1], ".", color="C3")
    sp = pts[~is_part_of_cell]
    plt.plot(sp[:, 0], sp[:, 1], ".", color="k")
    # plt.plot(pts[~is_inside, 0], pts[~is_part_of_cell, 1], ".", color="k")
    plt.triplot(pts[:, 0], pts[:, 1], cells)
    plt.axis("square")

    # show cells indices
    # for idx, barycenter in enumerate(np.sum(pts[cells], axis=1) / 3):
    #     plt.plot(*barycenter, "xk")
    #     plt.text(
    #         *barycenter, idx, horizontalalignment="center", verticalalignment="center"
    #     )

    # show node indices
    # for idx, pt in enumerate(pts):
    #     plt.text(
    #         *pt, idx, horizontalalignment="center", verticalalignment="center"
    #     )

    if full_screen:
        figManager = plt.get_current_fig_manager()
        try:
            figManager.window.showMaximized()
        except AttributeError:
            # Some backends have no window (e.g., Agg)
            pass

    if title is not None:
        plt.title(title)

    try:
        geo.show(level_set=False)
    except AttributeError:
        pass


def find_feature_points(geometries, num_steps: int = 10):
    n = len(geometries)

    # collect path pairs
    path_pairs = [
        [item0, item1]
        for i in range(n)
        for j in range(i + 1, n)
        for item0 in geometries[i].paths
        for item1 in geometries[j].paths
    ]

    points = np.column_stack(
        [
            _find_feature_points_between_two_paths(path0, path1, num_steps)
            for path0, path1 in path_pairs
        ]
    )

    if points.shape[1] > 0:
        points = unique_float_cols(points)

    return points.T


def _find_feature_points_between_two_paths(path0, path1, num_steps, nx=11, ny=11):
    """Given two geometries with their parametrization, this methods finds feature
    points, i.e., points where the boundaries meet. This is done by casting a net over
    the parameter space and performing `num_steps` Newton steps. Found solutions are
    checked for uniqueness.
    """
    # Throw a net
    t0, t1 = np.meshgrid(np.linspace(0.0, 1.0, nx), np.linspace(0.0, 1.0, ny))
    t = np.array([t0, t1]).reshape(2, -1)
    # t = np.random.rand(2, 100)

    tol = 1.0e-20

    # multi_newton(x0, is_inside, boundary_step, tol, max_num_steps=10):
    solutions = []
    for k in range(num_steps):
        f_t = path0.p(t[0]) - path1.p(t[1])

        # remove all inf values
        is_infinite = np.any(np.isinf(f_t), axis=0)
        if np.any(is_infinite):
            t = t[:, ~is_infinite]
            f_t = f_t[:, ~is_infinite]

        f_dot_f = np.einsum("ij,ij->j", f_t, f_t)
        is_sol = f_dot_f < tol

        if np.any(is_sol):
            solutions.append(t[:, is_sol])
            # remove all converged solutions
            t = t[:, ~is_sol]
            f_t = f_t[:, ~is_sol]

        jac_t = np.moveaxis(np.stack([path0.dp_dt(t[0]), -path1.dp_dt(t[1])]), 0, 1)

        # Kick out singular matrices
        det = jac_t[0, 0] * jac_t[1, 1] - jac_t[0, 1] * jac_t[1, 0]
        is_singular = np.abs(det) < 1.0e-13
        if np.any(is_singular):
            t = t[:, ~is_singular]
            f_t = f_t[:, ~is_singular]
            jac_t = jac_t[..., ~is_singular]

        # Simply make it explicitly.
        sols = []
        for k in range(f_t.shape[-1]):
            try:
                sols.append(np.linalg.solve(jac_t[..., k], f_t[:, k]))
            except np.linalg.linalg.LinAlgError:
                # singular matrix
                sols.append(np.zeros(f_t[:, k].shape))
        sols = np.array(sols).T

        # Newton step
        t -= sols

        # Kick out everything that leaves the unit square
        still_good = np.all((0.0 <= t) & (t <= 1.0), axis=0)
        t = t[:, still_good]

    if solutions:
        unique_sols = unique_float_cols(np.column_stack(solutions))
        points0 = path0.p(unique_sols[0])
        # points1 = path1.p(unique_sols[1])
    else:
        points0 = np.array([[], []])

    return points0


def unique_float_cols(data: np.ndarray, k: int = 0, tol: float = 1.0e-10):
    """In a (k, n) array `data`, find the unique columns."""
    if k == data.shape[0]:
        return data[:, 0]

    idx = np.argsort(data[k])
    data = data[:, idx]

    diff = data[k, 1:] - data[k, :-1]
    cut = diff > tol

    idx = np.where(cut)[0]
    chunks = np.split(data, idx + 1, axis=1)

    out = np.column_stack([unique_float_cols(chunk, k + 1, tol) for chunk in chunks])
    return out
