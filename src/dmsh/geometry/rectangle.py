import numpy as np

from .geometry import Geometry
from .polygon import LineSegmentPath


class Rectangle(Geometry):
    # One could simply make Rectangle a child class of Polygon. However, boundary steps
    # can be inaccurate for polygons (there is some computation involved).
    def __init__(self, x0, x1, y0, y1):
        assert x0 < x1
        assert y0 < y1
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.points = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        bounding_box = [
            np.min(self.points[:, 0]),
            np.max(self.points[:, 0]),
            np.min(self.points[:, 1]),
            np.max(self.points[:, 1]),
        ]
        self.paths = [
            LineSegmentPath(p0, p1)
            for p0, p1 in zip(self.points, np.roll(self.points, -1, axis=0))
        ]
        super().__init__(bounding_box, feature_points=self.points)

    def dist(self, x):
        # outside dist
        # https://gamedev.stackexchange.com/a/44496
        x = np.asarray(x)
        w = self.x1 - self.x0
        h = self.y1 - self.y0
        cx = (self.x0 + self.x1) / 2
        cy = (self.y0 + self.y1) / 2
        dx = np.abs(x[0] - cx) - w / 2
        dy = np.abs(x[1] - cy) - h / 2
        is_inside = (dx <= 0) & (dy <= 0)
        dx[dx < 0.0] = 0.0
        dy[dy < 0.0] = 0.0
        dist = np.sqrt(dx ** 2 + dy ** 2)
        # inside dist
        a = np.array(
            [
                x[0, is_inside] - self.x0,
                self.x1 - x[0, is_inside],
                x[1, is_inside] - self.y0,
                self.y1 - x[1, is_inside],
            ]
        )
        dist[is_inside] = -np.min(a, axis=0)
        return dist

    def boundary_step(self, x):
        x = np.asarray(x)
        assert x.shape[0] == 2

        is_one_dimensional = False
        if len(x.shape) == 1:
            is_one_dimensional = True
            x = x.reshape(-1, 1)

        cx = (self.x0 + self.x1) / 2
        cy = (self.y0 + self.y1) / 2
        w = self.x1 - self.x0
        h = self.y1 - self.y0

        X = x[0] - cx
        Y = x[1] - cy

        # Take care of the outside points
        X[X < -w / 2] = -w / 2
        X[X > +w / 2] = +w / 2
        Y[Y < -h / 2] = -h / 2
        Y[Y > +h / 2] = +h / 2

        # Interior points
        is_interior = (-w / 2 < X) & (X < w / 2) & (-h / 2 < Y) & (Y < h / 2)
        a = h * X < w * Y
        b = -h * X < w * Y
        Y[is_interior & a & b] = h / 2
        Y[is_interior & ~a & ~b] = -h / 2
        X[is_interior & ~a & b] = w / 2
        X[is_interior & a & ~b] = -w / 2

        X += cx
        Y += cy

        out = np.array([X, Y])
        if is_one_dimensional:
            out = out.reshape(-1)
        return out
