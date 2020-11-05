import numpy

from . import pypathlib


class LineSegmentPath:
    def __init__(self, x0, x1):
        """
        Initialize the x0 instance

        Args:
            self: (todo): write your description
            x0: (float): write your description
            x1: (int): write your description
        """
        self.x0 = x0
        self.x1 = x1
        return

    def p(self, t):
        """
        Multipy. numpy. t. t.

        Args:
            self: (todo): write your description
            t: (int): write your description
        """
        return numpy.multiply.outer(self.x0, 1 - t) + numpy.multiply.outer(self.x1, t)

    def dp_dt(self, t):
        """
        Return the time at t.

        Args:
            self: (todo): write your description
            t: (todo): write your description
        """
        ones = numpy.ones(t.shape)
        return numpy.multiply.outer(self.x0, -ones) + numpy.multiply.outer(
            self.x1, ones
        )


class Path:
    def __init__(self, points):
        """
        Initialize the bounding box.

        Args:
            self: (todo): write your description
            points: (todo): write your description
        """
        points = numpy.array(points)
        self.path = pypathlib.Path(points)
        self.bounding_box = [
            numpy.min(points[:, 0]),
            numpy.max(points[:, 0]),
            numpy.min(points[:, 1]),
            numpy.max(points[:, 1]),
        ]
        self.feature_points = points

        self.paths = [
            LineSegmentPath(p0, p1) for p0, p1 in zip(points[:-1], points[1:])
        ]
        return

    def dist(self, x):
        """
        Return the distance between x and x.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        return self.path.distance(x.T)

    def boundary_step(self, x):
        """
        Return the bounding step.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        return self.path.closest_points(x.T).T
