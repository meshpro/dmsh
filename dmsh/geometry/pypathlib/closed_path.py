import numpy

from .helpers import shoelace
from .path import Path


class ClosedPath(Path):
    def __init__(self, points):
        closed_points = numpy.concatenate([points, [points[0]]])
        super().__init__(closed_points)

        assert self.points.shape[0] > 2
        assert self.points.shape[1] == 2

        self.area = 0.5 * shoelace(self.points)
        self.positive_orientation = self.area >= 0
        if self.area < 0:
            self.area = -self.area

        self._is_convex_node = None
        return

    def signed_squared_distance(self, x):
        """Negative inside the polgon."""
        x = numpy.array(x)
        assert x.shape[1] == 2
        t, dist2, idx = self._all_distances(x)
        contains_points = self._contains_points(t, x, idx)
        dist2[contains_points] *= -1
        return dist2

    def signed_distance(self, x):
        """Negative inside the polgon."""
        x = numpy.array(x)
        assert x.shape[1] == 2
        t, dist2, idx = self._all_distances(x)
        dist = numpy.sqrt(dist2)
        contains_points = self._contains_points(t, x, idx)
        dist[contains_points] *= -1
        return dist

    def _contains_points(self, t, x, idx):
        r = numpy.arange(idx.shape[0])

        contains_points = numpy.zeros(x.shape[0], dtype=bool)

        pts0 = self.points
        pts1 = numpy.roll(self.points, -1, axis=0)

        # If the point is closest to a polygon edge, check which which side of the edge
        # it is on.
        is_closest_to_side = (0.0 < t[r, idx]) & (t[r, idx] < 1.0)
        tri = numpy.array(
            [
                x[is_closest_to_side],
                pts0[idx[is_closest_to_side]],
                pts1[idx[is_closest_to_side]],
            ]
        )

        contains_points[is_closest_to_side] = (
            shoelace(tri) > 0.0
        ) == self.positive_orientation

        # If the point is closest to a polygon node, check if the node is convex or
        # concave.
        is_closest_to_pt0 = t[r, idx] <= 0.0
        contains_points[is_closest_to_pt0] = ~self.is_convex_node[
            idx[is_closest_to_pt0]
        ]

        is_closest_to_pt1 = 1.0 <= t[r, idx]
        n = self.points.shape[0] - 1
        contains_points[is_closest_to_pt1] = ~self.is_convex_node[
            (idx[is_closest_to_pt1] + 1) % n
        ]

        return contains_points

    def contains_points(self, x, tol=1.0e-15):
        return self.signed_distance(x) < tol

    @property
    def is_convex_node(self):
        points = self.points[:-1]
        if self._is_convex_node is None:
            tri = numpy.array(
                [numpy.roll(points, +1, axis=0), points, numpy.roll(points, -1, axis=0)]
            )
            self._is_convex_node = numpy.equal(
                shoelace(tri) >= 0, self.positive_orientation
            )
        return self._is_convex_node
