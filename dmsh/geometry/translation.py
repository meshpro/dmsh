import numpy

from .geometry import Geometry


class Translation(Geometry):
    def __init__(self, geometry, v):
        """
        Initializes the bounding box.

        Args:
            self: (todo): write your description
            geometry: (todo): write your description
            v: (int): write your description
        """
        super().__init__()
        self.geometry = geometry
        self.v = v

        self.bounding_box = [
            geometry.bounding_box[0] + v[0],
            geometry.bounding_box[1] + v[0],
            geometry.bounding_box[2] + v[1],
            geometry.bounding_box[3] + v[1],
        ]
        self.feature_points = numpy.array([])
        return

    def dist(self, x):
        """
        Distribution distance between x and y.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        return self.geometry.dist((x.T - self.v).T)

    def boundary_step(self, x):
        """
        Boundary step.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return (self.geometry.boundary_step((x.T - self.v).T).T + self.v).T
