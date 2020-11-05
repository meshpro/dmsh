import numpy

from .geometry import Geometry


class Scaling(Geometry):
    def __init__(self, geometry, alpha):
        """
        Initializes the bounding box.

        Args:
            self: (todo): write your description
            geometry: (todo): write your description
            alpha: (float): write your description
        """
        super().__init__()
        self.geometry = geometry
        self.alpha = alpha
        self.bounding_box = alpha * numpy.array(geometry.bounding_box)
        self.feature_points = numpy.array([])

    def dist(self, x):
        """
        Return the distance between x and y.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        return self.geometry.dist(x / self.alpha)

    def boundary_step(self, x):
        """
        Boundary step at x.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return self.geometry.boundary_step(x / self.alpha) * self.alpha
