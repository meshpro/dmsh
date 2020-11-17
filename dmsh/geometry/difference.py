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
    def boundary_step(self, x, tol=1.0e-12, max_steps=100):
        # Scale the tolerance with the domain diameter. This is necessary at least for
        # polygons where the distance calculation is flawed with round-off proportional
        # to the edge lengths.
        try:
            tol *= self.geo0.diameter
        except AttributeError:
            pass

        alpha = numpy.array([self.geo0.dist(x), -self.geo1.dist(x)])
        mask = numpy.any(alpha > tol, axis=0) | numpy.all(alpha < -tol, axis=0)

        step = 0
        while numpy.any(mask):
            assert step <= max_steps, "Exceeded maximum number of boundary steps."
            step += 1

            x_tmp = x[:, mask]
            idx = numpy.argmax(alpha[:, mask], axis=0)
            if numpy.any(idx == 0):
                x_tmp[:, idx == 0] = self.geo0.boundary_step(x_tmp[:, idx == 0])
            if numpy.any(idx == 1):
                x_tmp[:, idx == 1] = self.geo1.boundary_step(x_tmp[:, idx == 1])
            x[:, mask] = x_tmp

            alpha = numpy.array([self.geo0.dist(x), -self.geo1.dist(x)])
            mask = numpy.any(alpha > tol, axis=0) | numpy.all(alpha < -tol, axis=0)
        return x
