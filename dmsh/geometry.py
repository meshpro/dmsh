# -*- coding: utf-8 -*-
#
import numpy


class Circle(object):
    def __init__(self, x0, r):
        self.x0 = x0
        self.r = r
        self.bounding_box = [x0[0] - r, x0[0] + r, x0[1] - r, x0[1] + r]
        return

    def isinside(self, x):
        return (x[0] - self.x0[0]) ** 2 + (x[1] - self.x0[1]) ** 2 - self.r ** 2

    def jac2(self, x):
        # TODO self.x0
        return numpy.array([
            4 * (x[0]**2 + x[1]**2 - self.r**2) * x[0],
            4 * (x[0]**2 + x[1]**2 - self.r**2) * x[1],
        ])

    def hessian2(self, x):
        # TODO self.x0
        return numpy.array([
            [8 * x[0]**2 + 4 * (x[0]**2 + x[1]**2 - self.r**2), 8 * x[0] * x[1]],
            [8 * x[0] * x[1], 8 * x[1]**2 + 4 * (x[0]**2 + x[1]**2 - self.r**2)],
        ])
