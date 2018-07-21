# -*- coding: utf-8 -*-
#
import numpy

from ..helpers import find_feature_points


class Difference(object):
    def __init__(self, geo0, geo1):
        self.geo0 = geo0
        self.geo1 = geo1
        self.bounding_box = geo0.bounding_box

        fp = [geo0.feature_points, geo1.feature_points]
        fp.append(find_feature_points([geo0, geo1]))
        self.feature_points = numpy.concatenate(fp)

        # Only keep the feature points on the outer boundary
        alpha = self.isinside(self.feature_points.T)
        tol = 1.0e-5
        is_on_boundary = (-tol < alpha) & (alpha < tol)
        self.feature_points = self.feature_points[is_on_boundary]
        return

    def plot(self):
        self.geo0.plot()
        self.geo1.plot()
        return

    def isinside(self, x):
        return numpy.max([self.geo0.isinside(x), -self.geo1.isinside(x)], axis=0)

    def boundary_step(self, x, tol=1.0e-12):
        # step for the is_inside with the smallest value
        alpha0 = self.geo0.isinside(x)
        alpha1 = self.geo1.isinside(x)
        while numpy.any(alpha0 > tol) or numpy.any(alpha1 < -tol):
            idx0 = alpha0 > tol
            if numpy.any(idx0):
                x[:, idx0] = self.geo0.boundary_step(x[:, idx0])
                alpha0 = self.geo0.isinside(x)
                continue

            idx1 = alpha1 < -tol
            if numpy.any(idx1):
                x[:, idx1] = self.geo1.boundary_step(x[:, idx1])
                alpha1 = self.geo1.isinside(x)
                continue
        return x
