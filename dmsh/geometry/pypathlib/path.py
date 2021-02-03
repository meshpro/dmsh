import numpy as np


class Path:
    def __init__(self, points):
        self.points = np.asarray(points)
        assert self.points.shape[1] == 2

        self.edges = self.points[1:] - self.points[:-1]
        self.e_dot_e = np.einsum("ij,ij->i", self.edges, self.edges)

        assert np.all(
            self.e_dot_e > 1.0e-12
        ), f"Edges of 0 length are not permitted (squared edge lengths: {self.e_dot_e})"

    def _all_distances(self, x):
        x = np.asarray(x)
        assert x.shape[1] == 2

        if self.points.shape[0] == 1:
            # In case there is only one point, i.e., no sides.
            diff = x[:, None] - self.points[None, :]
            dist2_points = np.einsum("ijk,ijk->ij", diff, diff)
            idx = np.zeros(dist2_points.shape[0], dtype=int)
            t = 0.0
            return t, dist2_points.T, idx

        # Find closest point for each side segment
        # <https://stackoverflow.com/q/51397389/353337>
        diff = x[:, None] - self.points[None, :-1]
        t = np.einsum("ijk,jk->ij", diff, self.edges) / self.e_dot_e
        t[t < 0.0] = 0.0
        t[t > 1.0] = 1.0

        # The squared distance from the point x to the infinite line defined by the
        # points x0, x1 (e = x1 - x0) is <proj - x, proj - x>, where proj is the
        # projection of x onto the line. The expression can be simplified to
        #
        #    (<e, e> <x-x0, x-x0> - <x-x0, e>**2) / <e, e>
        #
        # but this expression is numerically disadvantageous. (For example, the
        # expresison can become negative due to round-off.) Simply compute the
        # projection and the dot product.
        x_min_proj = diff - t[:, :, None] * self.edges[None, :, :]
        dist2_sides = np.einsum("ijk,ijk->ij", x_min_proj, x_min_proj)

        idx = np.argmin(dist2_sides, axis=1)
        dist2_sides = dist2_sides[np.arange(idx.shape[0]), idx]

        # t-parameter for each side, the squared min distance, and the index of the
        # closest side
        return t, dist2_sides, idx

    @property
    def diameter(self):
        # compute distance from all points to each other
        diff = self.points[:, None] - self.points[None, :]
        dist2 = np.einsum("ijk,ijk->ij", diff, diff)
        return np.sqrt(np.max(dist2))

    def squared_distance(self, x):
        """Get the squared distance of all points x to the polygon."""
        x = np.asarray(x)
        assert x.shape[1] == 2
        _, dist2_sides, _ = self._all_distances(x)
        return dist2_sides

    def distance(self, x):
        """Get the distance of all points x to the polygon."""
        return np.sqrt(self.squared_distance(x))

    def closest_points(self, x):
        """Get the closest points on the polygon."""
        x = np.asarray(x)
        assert x.shape[1] == 2
        t, _, idx = self._all_distances(x)

        pts0 = self.points[idx]
        pts1 = np.roll(self.points, -1, axis=0)[idx]
        r = np.arange(t.shape[0])
        t0 = t[r, idx]
        closest_points = (pts0.T * (1 - t0)).T + (pts1.T * t0).T
        return closest_points

    def plot(self, color="#1f77b4"):
        import matplotlib.pyplot as plt

        x = np.concatenate([self.points[:, 0], [self.points[0, 0]]])
        y = np.concatenate([self.points[:, 1], [self.points[0, 1]]])
        plt.plot(x, y, "-", color=color)

        plt.axis("square")

    def show(self, *args, **kwargs):
        import matplotlib.pyplot as plt

        self.plot(*args, **kwargs)
        plt.show()
