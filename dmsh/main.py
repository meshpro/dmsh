# -*- coding: utf-8 -*-
#
import fastfunc
import numpy
import scipy.spatial

from . import geometry
from .helpers import unique_rows, multi_newton, show


def element_size_function(x):
    return numpy.ones(x.shape[1])


def generate():
    h0 = 0.1
    f_scale = 1.2
    delta_t = 0.2

    geo = geometry.Circle([0.0, 0.0], 1.0)

    bb_width = geo.bounding_box[1] - geo.bounding_box[0]
    bb_height = geo.bounding_box[3] - geo.bounding_box[2]
    num_steps_x = int(bb_width / h0) + 1
    num_steps_y = int(bb_height / h0) + 1

    # Generate initial (staggered) point list from bounding box
    x, y = numpy.meshgrid(
        numpy.linspace(geo.bounding_box[0], geo.bounding_box[1], num_steps_x),
        numpy.linspace(geo.bounding_box[2], geo.bounding_box[3], num_steps_y),
    )
    # staggered
    x[1::2] += h0 / 2
    pts = numpy.array([x.reshape(-1), y.reshape(-1)])

    # remove points outside of the region
    pts = pts[:, geo.isinside(pts) < 0.0]

    # evaluate the element size function, remove points according to it
    p = element_size_function(pts)
    pts = pts[:, numpy.random.rand(pts.shape[1]) < p]

    while True:
        if True:
            # Re-Delauneyfication
            pts_old = pts.copy()

            # compute Delaunay triangulation
            tri = scipy.spatial.Delaunay(pts.T)
            cells = tri.simplices.copy()

            # kick out all cells whose barycenter is not in the region
            bc = numpy.sum(pts[:, cells], axis=-1) / 3.0
            cells = cells[geo.isinside(bc) < 0.0]

            # Find edges
            edges = numpy.concatenate(
                [cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]
            )
            edges = numpy.sort(edges, axis=1)
            edges, _, _ = unique_rows(edges)

        show(pts, cells, geo)

        edges_vec = pts[:, edges[:, 1]] - pts[:, edges[:, 0]]
        edge_lengths = numpy.sqrt(edges_vec[0] ** 2 + edges_vec[1] ** 2)

        # Evaluate element sizes at edge midpoints
        edge_midpoints = (pts[:, edges[:, 1]] + pts[:, edges[:, 0]]) / 2
        p = element_size_function(edge_midpoints)

        desired_lengths = (
            p
            * f_scale
            * numpy.sqrt(numpy.dot(edge_lengths, edge_lengths) / numpy.dot(p, p))
        )

        force_abs = desired_lengths - edge_lengths
        # only consider repulsive forces
        force_abs[force_abs < 0.0] = 0.0

        # force vectors
        force = force_abs * (edges_vec / edge_lengths)

        force_per_node = numpy.zeros(pts.T.shape)
        fastfunc.add.at(force_per_node, edges[:, 0], -force)
        fastfunc.add.at(force_per_node, edges[:, 1], +force)
        force_per_node = force_per_node.T

        pts = pts + delta_t * force_per_node

        # show(pts, cells, geo)

        # Some boundary points may have been pushed outside; bring them back onto the
        # boundary.
        is_outside = geo.isinside(pts) > 0.0
        pts[:, is_outside] = multi_newton(pts[:, is_outside], geo, tol=1.0e-10)

        # show(pts, cells, geo)

    exit(1)
    return None
