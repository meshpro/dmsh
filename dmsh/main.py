import numpy
import scipy.spatial

from .helpers import show as show_mesh
from .helpers import unique_rows


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


def generate(
    geo,
    edge_size,
    f_scale=1.2,
    delta_t=0.2,
    tol=1.0e-5,
    random_seed=0,
    show=False,
    max_steps=1000,
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

    num_feature_points = geo.feature_points.shape[0]
    if num_feature_points > 0:
        # remove all points which are equal to a feature point
        diff = numpy.array([[pt - fp for fp in geo.feature_points] for pt in pts])
        dist = numpy.einsum("...k,...k->...", diff, diff)
        ftol = h0 / 10
        equals_feature_point = numpy.any(dist < ftol ** 2, axis=1)
        pts = pts[~equals_feature_point]
        # Add feature points
        pts = numpy.concatenate([geo.feature_points, pts])

    cells, edges = _recell(pts, geo)
    pts_old = pts.copy()

    k = 0
    while True:
        if verbose:
            print("step {}".format(k))

        assert k <= max_steps, "Exceeded max_steps ({}).".format(max_steps)
        k += 1
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
        # more speed-up can be achieved if the weights where contiguous in memory, i.e.,
        # if force[k] was used
        n = pts.shape[0]
        force_per_node = numpy.array(
            [
                numpy.bincount(edges[:, 0], weights=-force[:, k], minlength=n)
                + numpy.bincount(edges[:, 1], weights=+force[:, k], minlength=n)
                for k in range(force.shape[1])
            ]
        ).T

        update = delta_t * force_per_node

        pts_old2 = pts.copy()

        pts[num_feature_points:] += update[num_feature_points:]

        # Some boundary points may have been pushed outside; bring them back onto the
        # boundary.
        is_outside = geo.dist(pts.T) > 0.0
        pts[is_outside] = geo.boundary_step(pts[is_outside].T).T

        diff = pts - pts_old2
        move2 = numpy.einsum("ij,ij->i", diff, diff)

        if verbose:
            print("max_move: {:.6e}".format(numpy.sqrt(numpy.max(move2))))

        if numpy.all(move2 < tol ** 2):
            break

    return pts, cells
