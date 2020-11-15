import math

import meshplex
import numpy
import scipy.spatial

from .helpers import show as show_mesh

# from .helpers import unique_rows


def _recell(pts, geo):
    # compute Delaunay triangulation
    tri = scipy.spatial.Delaunay(pts)
    cells = tri.simplices.copy()

    # kick out all cells whose barycenter is not in the geometry
    bc = numpy.sum(pts[cells], axis=1) / 3.0
    cells = cells[geo.dist(bc.T) < 0.0]

    # # kick out all cells whose barycenter or edge midpoints are not in the geometry
    # btol = 1.0e-3
    # bc = numpy.sum(pts[cells], axis=1) / 3.0
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


def create_staggered_grid(h, bounding_box):
    x_step = h
    y_step = h * numpy.sqrt(3) / 2
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
    x, y = numpy.meshgrid(
        midpoint[0] + x_step * numpy.arange(-x2, x2 + 1),
        midpoint[1] + y_step * numpy.arange(-y2, y2 + 1),
    )
    # Staggered, such that the midpoint is not moved.
    # Unconditionally move to the right, then add more points to the left.
    offset = (y2 + 1) % 2
    x[offset::2] += h / 2

    out = numpy.column_stack([x.reshape(-1), y.reshape(-1)])

    # add points in the staggered lines to preserve symmetry
    n = 2 * (-(-y2 // 2))
    extra = numpy.empty((n, 2))
    extra[:, 0] = midpoint[0] - x_step * x2 - h / 2
    extra[:, 1] = midpoint[1] + y_step * numpy.arange(-y2 + offset, y2 + 1, 2)

    out = numpy.concatenate([out, extra])
    return out


# def get_max_step(mesh):
#     # Some methods are stable (CPT), others can break down if the mesh isn't very
#     # smooth. A break-down manifests, for example, in a step size that lets triangles
#     # become fully flat or even "overshoot". After that, anything can happen. To prevent
#     # this, restrict the maximum step size to half of the minimum the incircle radius of
#     # all adjacent cells. This makes sure that triangles cannot "flip".
#     # <https://stackoverflow.com/a/57261082/353337>
#     max_step = numpy.full(mesh.points.shape[0], numpy.inf)
#     numpy.minimum.at(
#         max_step, mesh.cells["points"].reshape(-1), numpy.repeat(mesh.cell_inradius, 3),
#     )
#     max_step *= 0.5
#     return max_step


def generate(
    geo,
    edge_size,
    # smoothing_method="distmesh",
    tol=1.0e-5,
    random_seed=0,
    show=False,
    max_steps=10000,
    verbose=False,
):
    # Find h0 from edge_size (function)
    if callable(edge_size):
        edge_size_function = edge_size
        # Find h0 by sampling
        h00 = (geo.bounding_box[1] - geo.bounding_box[0]) / 100
        pts = create_staggered_grid(h00, geo.bounding_box)
        sizes = edge_size_function(pts.T)
        assert numpy.all(sizes > 0.0), "edge_size_function must be strictly positive."
        h0 = numpy.min(sizes)
    else:
        h0 = edge_size

        def edge_size_function(pts):
            return numpy.full(pts.shape[1], edge_size)

    if random_seed is not None:
        numpy.random.seed(random_seed)

    pts = create_staggered_grid(h0, geo.bounding_box)

    eps = 1.0e-10

    # remove points outside of the region
    pts = pts[geo.dist(pts.T) < eps]

    # evaluate the element size function, remove points according to it
    alpha = 1.0 / edge_size_function(pts.T) ** 2
    pts = pts[numpy.random.rand(pts.shape[0]) < alpha / numpy.max(alpha)]

    num_feature_points = len(geo.feature_points)
    if num_feature_points > 0:
        # remove all points which are equal to a feature point
        diff = numpy.array([[pt - fp for fp in geo.feature_points] for pt in pts])
        dist = numpy.einsum("...k,...k->...", diff, diff)
        ftol = h0 / 10
        equals_feature_point = numpy.any(dist < ftol ** 2, axis=1)
        pts = pts[~equals_feature_point]
        # Add feature points
        pts = numpy.concatenate([geo.feature_points, pts])

    cells = _recell(pts, geo)
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
    #         mesh.cells["points"],
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
        edge_size_function,
        max_steps,
        tol,
        verbose,
        show,
        delta_t=0.2,
        f_scale=1 + 0.4 / 2 ** (dim - 1),  # from the original article
    )
    points = mesh.points
    cells = mesh.cells["points"]

    return points, cells


def distmesh_smoothing(
    mesh,
    geo,
    num_feature_points,
    edge_size_function,
    max_steps,
    tol,
    verbose,
    show,
    delta_t,
    f_scale,
    # bad_cell_threshold=0.05,
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
            show_mesh(mesh.points, mesh.cells["points"], geo)

        edges = mesh.edges["points"]

        edges_vec = mesh.points[edges[:, 1]] - mesh.points[edges[:, 0]]
        edge_lengths = numpy.sqrt(numpy.einsum("ij,ij->i", edges_vec, edges_vec))
        edges_vec /= edge_lengths[..., None]

        # Evaluate element sizes at edge midpoints
        edge_midpoints = (mesh.points[edges[:, 1]] + mesh.points[edges[:, 0]]) / 2
        p = edge_size_function(edge_midpoints.T)
        desired_lengths = (
            f_scale
            * p
            * numpy.sqrt(numpy.dot(edge_lengths, edge_lengths) / numpy.dot(p, p))
        )

        force_abs = desired_lengths - edge_lengths
        # only consider repulsive forces
        force_abs[force_abs < 0.0] = 0.0

        # force vectors
        force = edges_vec * force_abs[..., None]

        # bincount replacement for the slow numpy.add.at
        # more speed-up can be achieved if the weights were contiguous in memory, i.e.,
        # if force[k] was used
        n = mesh.points.shape[0]
        force_per_point = numpy.array(
            [
                numpy.bincount(edges[:, 0], weights=-force[:, k], minlength=n)
                + numpy.bincount(edges[:, 1], weights=+force[:, k], minlength=n)
                for k in range(force.shape[1])
            ]
        ).T

        update = delta_t * force_per_point

        # # Limit the max step size to avoid overshoots
        # TODO this doesn't work for distmesh smoothing. hm.
        # mesh = meshplex.MeshTri(pts, cells)
        # max_step = get_max_step(mesh)
        # step_lengths = numpy.sqrt(numpy.einsum("ij,ij->i", update, update))
        # idx = step_lengths > max_step
        # update[idx] *= (max_step / step_lengths)[idx, None]
        # # alpha = numpy.min(max_step / step_lengths)
        # # update *= alpha

        # update coordinates
        points_new = mesh.points + update
        # leave feature points untouched
        points_new[:num_feature_points] = mesh.points[:num_feature_points]
        # Some mesh boundary points may have been moved off of the domain boundary,
        # either because they were pushed outside or because they just became boundary
        # points by way of cell removal. Move them all (back) onto the domain boundary.
        # is_outside = geo.dist(points_new.T) > 0.0
        # idx = is_outside
        # Alternative: Push all boundary points (the ones _inside_ the geometry as well)
        # back to the boundary.
        # idx = is_outside | is_boundary_point
        idx = mesh.is_boundary_point
        points_new[idx] = geo.boundary_step(points_new[idx].T).T

        # Make sure that the boundary step was performed correctly.
        print(geo.dist(points_new[idx].T))
        assert numpy.all(numpy.abs(geo.dist(points_new[idx].T)) < 1.0e-12)

        diff = points_new - mesh.points
        mesh.points = points_new

        print(geo.dist(mesh.points[mesh.is_boundary_point].T))

        # show_mesh(mesh.points, mesh.cells["points"], geo)
        mesh.show(mark_points=mesh.is_boundary_point)

        # We could do a _recell() here, but inverted boundary cell removal plus Lawson
        # flips produce the same result and are much cheaper. This is because, most of
        # the time, there are no cells to be removed and no edges to be flipped.
        # Because the rest of the algorithm is so cheap. The flip is still a fairly
        # expensive operation.
        # First kick out all boundary cells whose barycenters are not in the geometry.
        print("a", numpy.sum(mesh.is_boundary_point))
        mesh.remove_boundary_cells(
            lambda is_boundary_cell: geo.dist(
                mesh.compute_centroids(is_boundary_cell).T
            )
            > 0.0
        )
        print("b", numpy.sum(mesh.is_boundary_point))
        # mesh.remove_boundary_cells(
        #     lambda is_boundary_cell: mesh.compute_signed_cell_areas(is_boundary_cell)
        #     < 1.0e-10
        # )
        mesh.flip_until_delaunay()

        move2 = numpy.einsum("ij,ij->i", diff, diff)
        if verbose:
            print("max_move: {:.6e}".format(numpy.sqrt(numpy.max(move2))))
        if numpy.all(move2 < tol ** 2):
            break

    # print("num steps:  ", k)
    return mesh
