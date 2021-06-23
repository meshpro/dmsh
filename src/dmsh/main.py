import math
from typing import Callable, Union

import meshplex
import numpy as np
import scipy.spatial

from .helpers import show as show_mesh


def _create_cells(pts, geo):
    # compute Delaunay triangulation
    tri = scipy.spatial.Delaunay(pts)
    cells = tri.simplices.copy()

    # kick out all cells whose barycenter is not in the geometry
    bc = np.sum(pts[cells], axis=1) / 3.0
    cells = cells[geo.dist(bc.T) < 0.0]

    # # kick out all cells whose barycenter or edge midpoints are not in the geometry
    # btol = 1.0e-3
    # bc = np.sum(pts[cells], axis=1) / 3.0
    # barycenter_inside = geo.dist(bc.T) < btol
    # # Remove cells which are (partly) outside of the domain. Check at the midpoint of
    # # all edges.
    # mid0 = (pts[cells[:, 1]] + pts[cells[:, 2]]) / 2
    # mid1 = (pts[cells[:, 2]] + pts[cells[:, 0]]) / 2
    # mid2 = (pts[cells[:, 0]] + pts[cells[:, 1]]) / 2
    # edge_midpoints_inside = (
    #     (geo.dist(mid0.T) < btol)
    #     & (geo.dist(mid1.T) < btol)
    #     & (geo.dist(mid2.T) < btol)
    # )
    # cells = cells[barycenter_inside & edge_midpoints_inside]
    return cells


def _recell_and_boundary_step(mesh, geo, flip_tol):
    # We could do a _create_cells() here, but inverted boundary cell removal plus Lawson
    # flips produce the same result and are much cheaper. This is because, most of the
    # time, there are no cells to be removed and no edges to be flipped. (The flip is
    # still a fairly expensive operation.)
    while True:
        idx = mesh.is_boundary_point
        points_new = mesh.points.copy()
        points_new[idx] = geo.boundary_step(points_new[idx].T).T
        mesh.points = points_new
        #
        num_removed_cells = mesh.remove_boundary_cells(
            lambda is_bdry_cell: mesh.compute_signed_cell_volumes(is_bdry_cell)
            < 1.0e-10
        )
        #
        # The flip has to come right after the boundary cell removal to prevent
        # "degenerate cell" errors.
        mesh.flip_until_delaunay(tol=flip_tol)
        #
        if num_removed_cells == 0:
            break

    # Last kick out all boundary cells whose barycenters are not in the geometry.
    mesh.remove_boundary_cells(
        lambda is_bdry_cell: geo.dist(mesh.compute_cell_centroids(is_bdry_cell).T) > 0.0
    )


def create_staggered_grid(h, bounding_box):
    x_step = h
    y_step = h * np.sqrt(3) / 2
    bb_width = bounding_box[1] - bounding_box[0]
    bb_height = bounding_box[3] - bounding_box[2]
    midpoint = [
        (bounding_box[0] + bounding_box[1]) / 2,
        (bounding_box[2] + bounding_box[3]) / 2,
    ]

    num_x_steps = int(bb_width / x_step)
    if num_x_steps % 2 == 1:
        num_x_steps -= 1
    num_y_steps = int(bb_height / y_step)
    if num_y_steps % 2 == 1:
        num_y_steps -= 1

    # Generate initial (staggered) point list from bounding box.
    # Make sure that the midpoint is one point in the grid.
    x2 = num_x_steps // 2
    y2 = num_y_steps // 2
    x, y = np.meshgrid(
        midpoint[0] + x_step * np.arange(-x2, x2 + 1),
        midpoint[1] + y_step * np.arange(-y2, y2 + 1),
    )
    # Staggered, such that the midpoint is not moved.
    # Unconditionally move to the right, then add more points to the left.
    offset = (y2 + 1) % 2
    x[offset::2] += h / 2

    out = np.column_stack([x.reshape(-1), y.reshape(-1)])

    # add points in the staggered lines to preserve symmetry
    n = 2 * (-(-y2 // 2))
    extra = np.empty((n, 2))
    extra[:, 0] = midpoint[0] - x_step * x2 - h / 2
    extra[:, 1] = midpoint[1] + y_step * np.arange(-y2 + offset, y2 + 1, 2)

    out = np.concatenate([out, extra])
    return out


# def get_max_step(mesh):
#     # Some methods are stable (CPT), others can break down if the mesh isn't very
#     # smooth. A break-down manifests, for example, in a step size that lets triangles
#     # become fully flat or even "overshoot". After that, anything can happen. To prevent
#     # this, restrict the maximum step size to half of the minimum the incircle radius of
#     # all adjacent cells. This makes sure that triangles cannot "flip".
#     # <https://stackoverflow.com/a/57261082/353337>
#     max_step = np.full(mesh.points.shape[0], np.inf)
#     np.minimum.at(
#         max_step, mesh.cells("points").reshape(-1), np.repeat(mesh.cell_inradius, 3),
#     )
#     max_step *= 0.5
#     return max_step


def generate(
    geo,
    target_edge_size: Union[float, Callable],
    # smoothing_method="distmesh",
    tol: float = 1.0e-5,
    random_seed: int = 0,
    show: bool = False,
    max_steps: int = 10000,
    verbose: bool = False,
    flip_tol: float = 0.0,
):
    target_edge_size_function = (
        target_edge_size
        if callable(target_edge_size)
        else lambda pts: np.full(pts.shape[1], target_edge_size)
    )

    # Find h0 from edge_size (function)
    if callable(target_edge_size):
        # Find h0 by sampling
        h00 = (geo.bounding_box[1] - geo.bounding_box[0]) / 100
        pts = create_staggered_grid(h00, geo.bounding_box)
        sizes = target_edge_size_function(pts.T)
        assert np.all(
            sizes > 0.0
        ), "target_edge_size_function must be strictly positive."
        h0 = np.min(sizes)
    else:
        h0 = target_edge_size

    pts = create_staggered_grid(h0, geo.bounding_box)

    eps = 1.0e-10

    # remove points outside of the region
    pts = pts[geo.dist(pts.T) < eps]

    # evaluate the element size function, remove points according to it
    alpha = 1.0 / target_edge_size_function(pts.T) ** 2
    if random_seed is not None:
        np.random.seed(random_seed)
    pts = pts[np.random.rand(pts.shape[0]) < alpha / np.max(alpha)]

    num_feature_points = len(geo.feature_points)
    if num_feature_points > 0:
        # remove all points which are equal to a feature point
        diff = np.array([[pt - fp for fp in geo.feature_points] for pt in pts])
        dist = np.einsum("...k,...k->...", diff, diff)
        ftol = h0 / 10
        equals_feature_point = np.any(dist < ftol ** 2, axis=1)
        pts = pts[~equals_feature_point]
        # Add feature points
        pts = np.concatenate([geo.feature_points, pts])

    cells = _create_cells(pts, geo)
    mesh = meshplex.MeshTri(pts, cells)
    # When creating a mesh for the staggered grid, degenerate cells can very well occur
    # at the boundary, where points sit in a straight line. Remove those cells.
    mesh.remove_cells(mesh.q_radius_ratio < 1.0e-10)

    # # move boundary points to the boundary exactly
    # is_boundary_point = mesh.is_boundary_point.copy()
    # mesh.points[is_boundary_point] = geo.boundary_step(
    #     mesh.points[is_boundary_point].T
    # ).T

    # print(sum(is_boundary_point))
    # show_mesh(pts, cells, geo)
    # exit(1)

    # if smoothing_method == "odt":
    #     points, cells = optimesh.odt.fixed_point_uniform(
    #         mesh.points,
    #         mesh.cells("points"),
    #         max_num_steps=max_steps,
    #         verbose=verbose,
    #         boundary_step=geo.boundary_step,
    #     )
    # else:
    #     assert smoothing_method == "distmesh"
    dim = 2
    mesh = distmesh_smoothing(
        mesh,
        geo,
        num_feature_points,
        target_edge_size_function,
        max_steps,
        tol,
        verbose,
        show,
        delta_t=0.2,
        f_scale=1 + 0.4 / 2 ** (dim - 1),  # from the original article
        flip_tol=flip_tol,
    )
    points = mesh.points
    cells = mesh.cells("points")

    return points, cells


def distmesh_smoothing(
    mesh,
    geo,
    num_feature_points,
    target_edge_size_function,
    max_steps,
    tol,
    verbose,
    show,
    delta_t,
    f_scale,
    flip_tol=0.0,
):
    mesh.create_edges()

    k = 0
    move2 = [0.0]
    while True:
        # print()
        # print(f"step {k}")
        if verbose:
            print(f"step {k}")

        if k > max_steps:
            if verbose:
                print(f"Exceeded max_steps ({max_steps}).")
            break

        k += 1

        if show:
            print(f"max move: {math.sqrt(max(move2)):.3e}")
            show_mesh(mesh.points, mesh.cells("points"), geo)

        edges = mesh.edges["points"]

        edges_vec = mesh.points[edges[:, 1]] - mesh.points[edges[:, 0]]
        edge_lengths = np.sqrt(np.einsum("ij,ij->i", edges_vec, edges_vec))
        edges_vec /= edge_lengths[..., None]

        # Evaluate element sizes at edge midpoints
        edge_midpoints = (mesh.points[edges[:, 1]] + mesh.points[edges[:, 0]]) / 2
        p = target_edge_size_function(edge_midpoints.T)
        desired_lengths = (
            f_scale * p * np.sqrt(np.dot(edge_lengths, edge_lengths) / np.dot(p, p))
        )

        force_abs = desired_lengths - edge_lengths
        # only consider repulsive forces
        force_abs[force_abs < 0.0] = 0.0

        # force vectors
        force = edges_vec * force_abs[..., None]

        # bincount replacement for the slow np.add.at
        # more speed-up can be achieved if the weights were contiguous in memory, i.e.,
        # if force[k] was used
        n = mesh.points.shape[0]
        force_per_point = np.array(
            [
                np.bincount(edges[:, 0], weights=-force[:, k], minlength=n)
                + np.bincount(edges[:, 1], weights=+force[:, k], minlength=n)
                for k in range(force.shape[1])
            ]
        ).T

        update = delta_t * force_per_point

        # # Limit the max step size to avoid overshoots
        # TODO this doesn't work for distmesh smoothing. hm.
        # mesh = meshplex.MeshTri(pts, cells)
        # max_step = get_max_step(mesh)
        # step_lengths = np.sqrt(np.einsum("ij,ij->i", update, update))
        # idx = step_lengths > max_step
        # update[idx] *= (max_step / step_lengths)[idx, None]
        # # alpha = np.min(max_step / step_lengths)
        # # update *= alpha

        points_old = mesh.points.copy()

        # update coordinates
        points_new = mesh.points + update
        # leave feature points untouched
        points_new[:num_feature_points] = mesh.points[:num_feature_points]
        mesh.points = points_new
        # Some mesh boundary points may have been moved off of the domain boundary,
        # either because they were pushed outside or because they just became boundary
        # points by way of cell removal. Move them all (back) onto the domain boundary.
        # is_outside = geo.dist(points_new.T) > 0.0
        # idx = is_outside
        # Alternative: Push all boundary points (the ones _inside_ the geometry as well)
        # back to the boundary.
        # idx = is_outside | is_boundary_point
        _recell_and_boundary_step(mesh, geo, flip_tol)

        diff = points_new - points_old
        move2 = np.einsum("ij,ij->i", diff, diff)
        if verbose:
            print(f"max_move: {np.sqrt(np.max(move2)):.6e}")
        if np.all(move2 < tol ** 2):
            break

    # print("num steps:  ", k)

    # The cell removal steps in _recell_and_boundary_step() might create points which
    # aren't part of any cell (dangling points). Remove them now.
    mesh.remove_dangling_points()
    return mesh
