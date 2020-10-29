import numpy

from . import pypathlib
from .geometry import Geometry


class LineSegmentPath:
    def __init__(self, x0, x1):
        self.x0 = x0
        self.x1 = x1

    def p(self, t):
        return numpy.multiply.outer(self.x0, 1 - t) + numpy.multiply.outer(self.x1, t)

    def dp_dt(self, t):
        ones = numpy.ones(t.shape)
        return numpy.multiply.outer(self.x0, -ones) + numpy.multiply.outer(
            self.x1, ones
        )


class Polygon(Geometry):
    def __init__(self, points):
        super().__init__()
        points = numpy.asarray(points)
        self.bounding_box = [
            numpy.min(points[:, 0]),
            numpy.max(points[:, 0]),
            numpy.min(points[:, 1]),
            numpy.max(points[:, 1]),
        ]
        self.polygon = pypathlib.ClosedPath(points)
        self.feature_points = points
        self.paths = [
            LineSegmentPath(p0, p1)
            for p0, p1 in zip(points, numpy.roll(points, -1, axis=0))
        ]
        self.diameter = self.polygon.diameter

    def dist(self, x):
        assert x.shape[0] == 2
        X = x.reshape(2, -1)
        out = self.polygon.signed_distance(X.T)
        return out.reshape(x.shape[1:])

    def boundary_step(self, x):
        return self.polygon.closest_points(x.T).T

    def plot(self, level_set=True):
        import matplotlib.pyplot as plt

        if level_set:
            self._plot_level_set()

        obj = plt.Polygon(self.feature_points, color="k", fill=False)
        plt.gca().add_artist(obj)

        plt.gca().set_aspect("equal")
