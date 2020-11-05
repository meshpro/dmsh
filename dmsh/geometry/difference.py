import numpy

from ..helpers import find_feature_points
from .geometry import Geometry


class Difference(Geometry):
    def __init__(self, geo0, geo1):
        """
        Initialize a bounding box.

        Args:
            self: (todo): write your description
            geo0: (list): write your description
            geo1: (list): write your description
        """
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

    def dist(self, x):
        """
        Return the distance between x and x.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        return numpy.max([self.geo0.dist(x), -self.geo1.dist(x)], axis=0)

    # Choose tolerance above sqrt(machine_eps). This is necessary as the polygon
    # dist() is only accurate to that precision.
    def boundary_step(self, x, tol=1.0e-7, max_steps=100):
        """
        Finds the boundary of x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
            tol: (float): write your description
            max_steps: (int): write your description
        """
        # Scale the tolerance with the domain diameter. This is necessary at least for
        # polygons where the distance calculation is flawed with round-off proportional
        # to the edge lengths.
        try:
            tol *= self.geo0.diameter
        except AttributeError:
            pass

        # step for the is_inside with the smallest value
        idx0 = self.geo0.dist(x) > tol
        idx1 = self.geo1.dist(x) < -tol

        k = 0
        while numpy.any(idx0) or numpy.any(idx1):
            assert k <= max_steps, "Exceeded maximum number of boundary steps."
            k += 1
            if numpy.any(idx0):
                x[:, idx0] = self.geo0.boundary_step(x[:, idx0])
            if numpy.any(idx1):
                x[:, idx1] = self.geo1.boundary_step(x[:, idx1])

            idx0 = self.geo0.dist(x) > tol
            idx1 = self.geo1.dist(x) < -tol
        return x
