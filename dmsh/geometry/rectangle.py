import numpy
import matplotlib.pyplot as plt

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
        self.points = numpy.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        self.bounding_box = [
            numpy.min(self.points[:, 0]),
            numpy.max(self.points[:, 0]),
            numpy.min(self.points[:, 1]),
            numpy.max(self.points[:, 1]),
        ]
        self.feature_points = self.points
        self.paths = [
            LineSegmentPath(p0, p1)
            for p0, p1 in zip(self.points, numpy.roll(self.points, -1, axis=0))
        ]

    def plot(self):
        x, y = numpy.concatenate([self.points, [self.points[0]]]).T
        plt.gca().set_aspect("equal")
        plt.plot(x, y, "-", color="C0")

    def dist(self, x):
        # outside dist
        # https://gamedev.stackexchange.com/a/44496
        w = self.x1 - self.x0
        h = self.y1 - self.y0
        cx = (self.x0 + self.x1) / 2
        cy = (self.y0 + self.y1) / 2
        dx = numpy.abs(x[0] - cx) - w / 2
        dy = numpy.abs(x[1] - cy) - h / 2
        is_inside = (dx <= 0) & (dy <= 0)
        dx[dx < 0.0] = 0.0
        dy[dy < 0.0] = 0.0
        dist = numpy.sqrt(dx ** 2 + dy ** 2)
        # inside dist
        a = numpy.array([
            x[0, is_inside] - self.x0,
            self.x1 - x[0, is_inside],
            x[1, is_inside] - self.y0,
            self.y1 - x[1, is_inside],
        ])
        dist[is_inside] = -numpy.min(a, axis=0)
        return dist

    def boundary_step(self, x):
        out = x.copy()
        out[0][x[0] < self.x0] = self.x0
        out[0][x[0] > self.x1] = self.x1
        out[1][x[1] < self.y0] = self.y0
        out[1][x[1] > self.y1] = self.y1
        return out
