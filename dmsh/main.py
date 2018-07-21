# -*- coding: utf-8 -*-
#
import fastfunc
import numpy
import scipy.spatial

from .helpers import unique_rows, show as show_mesh


def homogenous(x):
    return numpy.ones(x.shape[0])


def _recell(pts, geo):
    # compute Delaunay triangulation
    tri = scipy.spatial.Delaunay(pts)
    cells = tri.simplices.copy()

    # kick out all cells whose barycenter is not in the geometry
    bc = numpy.sum(pts[cells], axis=1) / 3.0
    cells = cells[geo.dist(bc.T) < 0.0]

    # Determine edges
    edges = numpy.concatenate([cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]])
    edges = numpy.sort(edges, axis=1)
    edges, _, _ = unique_rows(edges)
    return cells, edges


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

    # Generate initial (staggered) point list from bounding box
    # Make the meshgrid symmetric around the bounding box midpoint
    x, y = numpy.meshgrid(
        midpoint[0] + x_step * numpy.arange(-num_x_steps // 2, num_x_steps // 2 + 1),
        midpoint[1] + y_step * numpy.arange(-num_y_steps // 2, num_y_steps // 2 + 1),
    )
    # staggered, such that the midpoint is not moved
    x[num_x_steps % 2 :: 2] += h / 2
    return numpy.column_stack([x.reshape(-1), y.reshape(-1)])


def generate(geo, edge_size, f_scale=1.2, delta_t=0.2, show=False):
    # Find h0 from edge_size (function)
    if callable(edge_size):
        edge_size_function = edge_size
        # Find h0 by sampling
        h00 = (geo.bounding_box[1] - geo.bounding_box[0]) / 100
        pts = create_staggered_grid(h00, geo.bounding_box)
        h0 = numpy.min(edge_size_function(pts.T))
    else:
        h0 = edge_size

        def edge_size_function(pts):
            return numpy.full(pts.shape[1], edge_size)

    pts = create_staggered_grid(h0, geo.bounding_box)

    eps = 1.0e-10

    # remove points outside of the region
    pts = pts[geo.dist(pts.T) < eps]

    # evaluate the element size function, remove points according to it
    alpha = 1.0 / edge_size_function(pts.T) ** 2
    pts = pts[numpy.random.rand(pts.shape[0]) < alpha / numpy.max(alpha)]

    # remove all points which are equal to a feature point
    diff = numpy.array([
        [pt - fp for fp in geo.feature_points]
        for pt in pts
    ])
    dist = numpy.einsum("...k,...k->...", diff, diff)
    tol = h0 / 10
    equals_feature_point = numpy.any(dist < tol**2, axis=1)
    pts = pts[~equals_feature_point]

    # Add feature points
    num_feature_points = geo.feature_points.shape[0]
    if num_feature_points > 0:
        pts = numpy.concatenate([geo.feature_points, pts])

    cells, edges = _recell(pts, geo)
    pts_old = pts.copy()

    while True:
        diff = pts - pts_old
        move2 = numpy.einsum("ij,ij->i", diff, diff)
        if numpy.any(move2 > 1.0e-2 ** 2):
            pts_old = pts.copy()
            cells, edges = _recell(pts, geo)

        if show:
            show_mesh(pts, cells, geo)

        edges_vec = pts[edges[:, 1]] - pts[edges[:, 0]]
        edge_lengths = numpy.sqrt(numpy.einsum("ij,ij->i", edges_vec, edges_vec))
        edges_vec /= edge_lengths[..., None]

        # Evaluate element sizes at edge midpoints
        edge_midpoints = (pts[edges[:, 1]] + pts[edges[:, 0]]) / 2
        desired_lengths = f_scale * edge_size_function(edge_midpoints.T)

        force_abs = desired_lengths - edge_lengths
        # only consider repulsive forces
        force_abs[force_abs < 0.0] = 0.0

        # force vectors
        force = edges_vec * force_abs[..., None]

        force_per_node = numpy.zeros(pts.shape)
        fastfunc.add.at(force_per_node, edges[:, 0], -force)
        fastfunc.add.at(force_per_node, edges[:, 1], +force)

        update = delta_t * force_per_node

        pts_old2 = pts.copy()

        pts[num_feature_points:] += update[num_feature_points:]
        # Some boundary points may have been pushed outside; bring them back onto the
        # boundary.
        is_outside = geo.dist(pts.T) > 0.0
        pts[is_outside] = geo.boundary_step(pts[is_outside].T).T

        diff = pts - pts_old2
        move2 = numpy.einsum("ij,ij->i", diff, diff)
        if numpy.all(move2 < 1.0e-5 ** 2):
            break

    return pts, cells
