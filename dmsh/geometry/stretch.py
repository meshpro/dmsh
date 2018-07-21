# -*- coding: utf-8 -*-
#
import numpy


class Stretch(object):
    def __init__(self, geometry, v):
        self.geometry = geometry
        self.alpha = numpy.sqrt(numpy.dot(v, v))
        self.v = v / self.alpha

        # bounding box
        bb = geometry.bounding_box
        corners = numpy.array(
            [[bb[0], bb[2]], [bb[1], bb[2]], [bb[1], bb[3]], [bb[0], bb[3]]]
        )
        vx = numpy.multiply.outer(numpy.dot(self.v, corners.T), self.v)
        stretched_corners = (vx * self.alpha + (corners - vx)).T
        self.bounding_box = [
            numpy.min(stretched_corners[0]),
            numpy.max(stretched_corners[0]),
            numpy.min(stretched_corners[1]),
            numpy.max(stretched_corners[1]),
        ]
        self.feature_points = numpy.array([])
        return

    def plot(self):
        return

    def isinside(self, x):
        # scale the component of x in direction v by 1/alpha
        vx = numpy.multiply.outer(numpy.dot(self.v, x), self.v)
        y = vx / self.alpha + (x.T - vx)
        return self.geometry.isinside(y.T)

    def boundary_step(self, x):
        vx = numpy.multiply.outer(numpy.dot(self.v, x), self.v)
        y = vx / self.alpha + (x.T - vx)
        y2 = self.geometry.boundary_step(y.T)
        vy2 = numpy.multiply.outer(numpy.dot(self.v, y2), self.v)
        return (vy2 * self.alpha + (y2.T - vy2)).T
