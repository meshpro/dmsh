import numpy


class Path:
    def __init__(self, points):
        self.points = numpy.asarray(points)
        assert self.points.shape[1] == 2

        self.edges = self.points[1:] - self.points[:-1]
        self.e_dot_e = numpy.einsum("ij,ij->i", self.edges, self.edges)

        assert numpy.all(
            self.e_dot_e > 1.0e-12
        ), "Edges of 0 length are not permitted (edge lengths: {})".format(
            numpy.sqrt(self.e_dot_e)
        )

    def _all_distances(self, x):
        x = numpy.asarray(x)
        assert x.shape[1] == 2

        # Find closest point for each side segment
        # <https://stackoverflow.com/q/51397389/353337>
        diff = x[:, None, :] - self.points[None, :, :]
        diff_dot_edge = numpy.einsum("ijk,jk->ij", diff[:, :-1], self.edges)
        t = diff_dot_edge / self.e_dot_e

        # The squared distance from the point x to the line defined by the points x0, x1
        # is
        #
        # (<x1-x0, x1-x0> <x-x0, x-x0> - <x-x0, x1-x0>**2) / <x1-x0, x1-x0>
        #
        dist2_points = numpy.einsum("ijk,ijk->ij", diff, diff)
        dist2_sides = (
            self.e_dot_e * dist2_points[:, :-1] - diff_dot_edge ** 2
        ) / self.e_dot_e
        # Wipe out small negative values
        dist2_sides = numpy.maximum(dist2_sides, numpy.zeros(dist2_sides.shape))

        # The numerator can be written as
        #
        #   (<x1-x0, x1-x0> <x-x0, x-x0> - <x-x0, x1-x0>**2)
        #   = 1/2 * sum ((x1-x0) (x-x0).T - (x-x0) (x1-x0).T)**2,
        #
        # (Lagrange' identity, <https://en.wikipedia.org/wiki/Lagrange%27s_identity>)
        # which is numerically guaranteed to be positive.
        #
        # <https://math.stackexchange.com/q/2856409/36678>
        # a = numpy.einsum("jk,ijl->ijkl", self.edges, diff)
        # b = a - numpy.moveaxis(a, 2, 3)
        # dist2_sides2 = 0.5 * numpy.einsum("ijkl,ijkl->ij", b, b)

        # Get the squared distance to the polygon. By default equals the distance to the
        # line, unless t < 0 (then the squared distance to x0), unless t > 1 (then the
        # squared distance to x1).
        t0 = t < 0.0
        t1 = t > 1.0
        t[t0] = 0.0
        t[t1] = 1.0
        dist2_sides[t0] = dist2_points[:, :-1][t0]
        dist2_sides[t1] = dist2_points[:, 1:][t1]

        if dist2_sides.shape[1] > 0:
            idx = numpy.argmin(dist2_sides, axis=1)
            dist2_sides = dist2_sides[numpy.arange(idx.shape[0]), idx]
        else:
            idx = numpy.zeros(dist2_points.shape[0], dtype=int)
            dist2_sides = dist2_points[:, 0]

        return t, dist2_sides, idx

    @property
    def diameter(self):
        # compute distance from all points to each other
        diff = self.points[:, None] - self.points[None, :]
        dist2 = numpy.einsum("ijk,ijk->ij", diff, diff)
        return numpy.sqrt(numpy.max(dist2))

    def squared_distance(self, x):
        """Get the squared distance of all points x to the polygon `poly`.
        """
        x = numpy.asarray(x)
        assert x.shape[1] == 2
        _, dist2_sides, _ = self._all_distances(x)
        return dist2_sides

    def distance(self, x):
        """Get the distance of all points x to the polygon `poly`.
        """
        return numpy.sqrt(self.squared_distance(x))

    def closest_points(self, x):
        """Get the closest points on the polygon.
        """
        x = numpy.asarray(x)
        assert x.shape[1] == 2
        t, _, idx = self._all_distances(x)

        pts0 = self.points[idx]
        pts1 = numpy.roll(self.points, -1, axis=0)[idx]
        r = numpy.arange(t.shape[0])
        t0 = t[r, idx]
        closest_points = (pts0.T * (1 - t0)).T + (pts1.T * t0).T
        return closest_points

    def plot(self, color="#1f77b4"):
        import matplotlib.pyplot as plt

        x = numpy.concatenate([self.points[:, 0], [self.points[0, 0]]])
        y = numpy.concatenate([self.points[:, 1], [self.points[0, 1]]])
        plt.plot(x, y, "-", color=color)

        plt.axis("square")

    def show(self, *args, **kwargs):
        import matplotlib.pyplot as plt

        self.plot(*args, **kwargs)
        plt.show()
