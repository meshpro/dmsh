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
    h0 = 0.5
    f_scale = 1.2
    delta_t = 0.2

    geo = geometry.Circle([0.0, 0.0], 1.0)

    x_step = h0
    y_step = h0 * numpy.sqrt(3) / 2
    bb_width = geo.bounding_box[1] - geo.bounding_box[0]
    bb_height = geo.bounding_box[3] - geo.bounding_box[2]
    midpoint = [
        (geo.bounding_box[0] + geo.bounding_box[1]) / 2,
        (geo.bounding_box[2] + geo.bounding_box[3]) / 2,
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
    # staggered
    x[1::2] += h0 / 2
    pts = numpy.array([x.reshape(-1), y.reshape(-1)])

    eps = 1.0e-10

    # remove points outside of the region
    pts = pts[:, geo.isinside(pts) < eps]

    # evaluate the element size function, remove points according to it
    p = element_size_function(pts)
    pts = pts[:, numpy.random.rand(pts.shape[1]) < p]

    # import matplotlib.pyplot as plt
    # t = numpy.linspace(0.0, 2 * numpy.pi, 100)
    # plt.plot(numpy.cos(t), numpy.sin(t), '-')
    # plt.plot(pts[0], pts[1], '.')
    # plt.axis("square")
    # plt.show()
    # exit(1)

    while True:
        if True:
            # Re-Delauneyfication
            # pts_old = pts.copy()

            # compute Delaunay triangulation
            tri = scipy.spatial.Delaunay(pts.T)
            cells = tri.simplices.copy()

            # kick out all cells whose barycenter is not in the geometry
            bc = numpy.sum(pts[:, cells], axis=-1) / 3.0
            cells = cells[geo.isinside(bc) < 0.0]

            # Determine edges
            edges = numpy.concatenate(
                [cells[:, [0, 1]], cells[:, [1, 2]], cells[:, [2, 0]]]
            )
            edges = numpy.sort(edges, axis=1)
            edges, _, _ = unique_rows(edges)

        show(pts, cells, geo)

        edges_vec = pts[:, edges[:, 1]] - pts[:, edges[:, 0]]
        edge_lengths = numpy.sqrt(edges_vec[0] ** 2 + edges_vec[1] ** 2)
        edges_vec /= edge_lengths

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
        force = force_abs * edges_vec

        force_per_node = numpy.zeros(pts.T.shape)
        fastfunc.add.at(force_per_node, edges[:, 0], -force)
        fastfunc.add.at(force_per_node, edges[:, 1], +force)

        print(pts[:, 7])
        pts += delta_t * force_per_node.T
        print(pts[:, 7])
        exit(1)

        # show(pts, cells, geo)

        # Some boundary points may have been pushed outside; bring them back onto the
        # boundary.
        is_outside = geo.isinside(pts) > 0.0
        pts[:, is_outside] = multi_newton(pts[:, is_outside], geo, tol=1.0e-10)

        # show(pts, cells, geo)

    exit(1)
    return None
