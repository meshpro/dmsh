# -*- coding: utf-8 -*-
#
import numpy
import polypy


class LineSegmentPath(object):
    def __init__(self, x0, x1):
        self.x0 = x0
        self.x1 = x1
        return

    def p(self, t):
        return numpy.multiply.outer(self.x0, 1 - t) + numpy.multiply.outer(self.x1, t)

    def dp_dt(self, t):
        ones = numpy.ones(t.shape)
        return numpy.multiply.outer(self.x0, -ones) + numpy.multiply.outer(
            self.x1, ones
        )


class Polygon(object):
    def __init__(self, points):
        points = numpy.array(points)
        self.bounding_box = [
            numpy.min(points[:, 0]),
            numpy.max(points[:, 0]),
            numpy.min(points[:, 1]),
            numpy.max(points[:, 1]),
        ]
        self.polygon = polypy.Polygon(points)
        self.feature_points = points

        self.paths = [
            LineSegmentPath(p0, p1)
            for p0, p1 in zip(points, numpy.roll(points, -1, axis=0))
        ]
        return

    def plot(self):
        self.polygon.plot(color="#1f77b4")
        return

    def isinside(self, x):
        return self.polygon.signed_squared_distance(x.T)

    def boundary_step(self, x):
        return self.polygon.closest_points(x.T).T
