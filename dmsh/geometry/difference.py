import numpy

from ..helpers import find_feature_points
from .geometry import Geometry


class Difference(Geometry):
    def __init__(self, geo0, geo1):
        super().__init__()
        self.geo0 = geo0
        self.geo1 = geo1
        self.bounding_box = geo0.bounding_box

        fp = [geo0.feature_points, geo1.feature_points]
        fp.append(find_feature_points([geo0, geo1]))
        self.feature_points = numpy.concatenate(fp)

        # Only keep the feature points on the outer boundary
        alpha = self.dist(self.feature_points.T)
        tol = 1.0e-5
        is_on_boundary = (-tol < alpha) & (alpha < tol)
        self.feature_points = self.feature_points[is_on_boundary]

        self.paths = [path for geo in [geo0, geo1] for path in geo.paths]

    def dist(self, x):
        return numpy.max([self.geo0.dist(x), -self.geo1.dist(x)], axis=0)

    # Choose tolerance above sqrt(machine_eps). This is necessary as the polygon
    # dist() is only accurate to that precision.
    def boundary_step(self, x, tol=1.0e-7, max_steps=100):
        # Scale the tolerance with the domain diameter. This is necessary at least for
        # polygons where the distance calculation is flawed with round-off proportional
        # to the edge lengths.
        try:
            tol *= self.geo0.diameter
        except AttributeError:
            pass

        # step for the is_inside with the smallest value
        gd0 = self.geo0.dist(x)
        gd1 = self.geo1.dist(x)
        geo0_out = gd0 > tol
        geo0_on = (gd0 < tol) & (gd0 > -tol)
        geo0_in = gd0 < -tol
        geo1_out = gd1 > tol
        geo1_in = gd1 < -tol

        k = 0
        while numpy.any(numpy.abs(numpy.max([gd0, -gd1], axis=0)) > tol):
            assert k <= max_steps, "Exceeded maximum number of boundary steps."
            k += 1

            idx = geo0_out
            if numpy.any(geo0_out):
                x[:, idx] = self.geo0.boundary_step(x[:, idx])

            idx = (geo0_on | geo0_in) & geo1_in
            if numpy.any(idx):
                x[:, idx] = self.geo1.boundary_step(x[:, idx])

            idx = geo0_in & geo1_out
            if numpy.any(idx):
                closer0 = numpy.abs(gd0) < numpy.abs(gd1)
                x[:, idx & closer0] = self.geo0.boundary_step(x[:, idx & closer0])
                x[:, idx & ~closer0] = self.geo1.boundary_step(x[:, idx & ~closer0])

            gd0 = self.geo0.dist(x)
            gd1 = self.geo1.dist(x)
            geo0_out = gd0 > tol
            geo0_on = (gd0 < tol) & (gd0 > -tol)
            geo0_in = gd0 < -tol
            geo1_out = gd1 > tol
            geo1_in = gd1 < -tol
        return x
