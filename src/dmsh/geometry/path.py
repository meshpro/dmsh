import numpy as np

from . import pypathlib


class LineSegmentPath:
    def __init__(self, x0, x1):
        self.x0 = x0
        self.x1 = x1
        return

    def p(self, t):
        return np.multiply.outer(self.x0, 1 - t) + np.multiply.outer(self.x1, t)

    def dp_dt(self, t):
        ones = np.ones(t.shape)
        return np.multiply.outer(self.x0, -ones) + np.multiply.outer(self.x1, ones)


class Path:
    def __init__(self, points):
        points = np.array(points)
        self.path = pypathlib.Path(points)
        self.bounding_box = [
            np.min(points[:, 0]),
            np.max(points[:, 0]),
            np.min(points[:, 1]),
            np.max(points[:, 1]),
        ]
        self.feature_points = points

        self.paths = [
            LineSegmentPath(p0, p1) for p0, p1 in zip(points[:-1], points[1:])
        ]
        return

    def dist(self, x):
        return self.path.distance(x.T)

    def boundary_step(self, x):
        return self.path.closest_points(x.T).T
