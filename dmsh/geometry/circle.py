import numpy

from .geometry import Geometry


class CirclePath:
    def __init__(self, x0, r):
        """
        Initialize the x0.

        Args:
            self: (todo): write your description
            x0: (float): write your description
            r: (int): write your description
        """
        self.x0 = x0
        self.r = r

    def p(self, t):
        """
        Name : math : \ r_ { t }

        Args:
            self: (todo): write your description
            t: (int): write your description
        """
        v = numpy.array([numpy.cos(2 * numpy.pi * t), numpy.sin(2 * numpy.pi * t)])
        return ((self.r * v).T + self.x0).T

    def dp_dt(self, t):
        """
        Return the time at time t

        Args:
            self: (todo): write your description
            t: (todo): write your description
        """
        return (
            self.r
            * 2
            * numpy.pi
            * numpy.array([-numpy.sin(2 * numpy.pi * t), numpy.cos(2 * numpy.pi * t)])
        )


class Circle(Geometry):
    def __init__(self, x0, r):
        """
        Initializes the bounding box.

        Args:
            self: (todo): write your description
            x0: (float): write your description
            r: (int): write your description
        """
        super().__init__()
        self.x0 = x0
        self.r = r
        self.bounding_box = [x0[0] - r, x0[0] + r, x0[1] - r, x0[1] + r]
        self.paths = [CirclePath(x0, r)]
        self.feature_points = numpy.array([[], []]).T

    def dist(self, x):
        """
        Compute the distance between x and y.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        assert x.shape[0] == 2
        y = (x.T - self.x0).T
        return numpy.sqrt(numpy.einsum("i...,i...->...", y, y)) - self.r

    def boundary_step(self, x):
        """
        Boundaryary step.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        # simply project onto the circle
        y = (x.T - self.x0).T
        r = numpy.sqrt(numpy.einsum("ij,ij->j", y, y))
        return ((y / r * self.r).T + self.x0).T

    def plot(self, level_set=True):
        """
        Plot the contour.

        Args:
            self: (todo): write your description
            level_set: (todo): write your description
        """
        import matplotlib.pyplot as plt

        if level_set:
            X, Y, Z = self._get_xyz()
            alpha = numpy.max(numpy.abs(Z))
            cf = plt.contourf(
                X, Y, Z, levels=20, cmap=plt.cm.coolwarm, vmin=-alpha, vmax=alpha
            )
            plt.colorbar(cf)

        circle1 = plt.Circle(self.x0, self.r, color="k", fill=False)
        plt.gca().add_artist(circle1)

        plt.gca().set_aspect("equal")
