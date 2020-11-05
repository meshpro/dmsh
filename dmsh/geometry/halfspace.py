import numpy

from .geometry import Geometry


class LinePath:
    def __init__(self, v, tangent):
        """
        Initialize the tang variable.

        Args:
            self: (todo): write your description
            v: (int): write your description
            tangent: (todo): write your description
        """
        super().__init__()
        self.v = v
        self.tangent = tangent

    def p(self, t):
        """This parametrization of the line is (inf, inf) for t=0 and t=1."""
        # Don't warn on division by 0
        with numpy.errstate(divide="ignore"):
            out = (
                numpy.multiply.outer(self.tangent, (2 * t - 1) / t / (1 - t)).T + self.v
            ).T
        return out

    def dp_dt(self, t):
        """
        Return the time at t.

        Args:
            self: (todo): write your description
            t: (todo): write your description
        """
        with numpy.errstate(divide="ignore"):
            dt = 1 / t ** 2 + 1 / (1 - t) ** 2
        return numpy.multiply.outer(self.tangent, dt)


class HalfSpace(Geometry):
    def __init__(self, normal, alpha):
        """
        Initialize the bounding box.

        Args:
            self: (todo): write your description
            normal: (todo): write your description
            alpha: (float): write your description
        """
        super().__init__()
        self.normal = normal
        self.alpha = alpha

        self.bounding_box = [-numpy.inf, +numpy.inf, -numpy.inf, +numpy.inf]
        self.feature_points = numpy.array([])

        # One point on the line:
        v = self.normal / numpy.dot(self.normal, self.normal) * self.alpha
        tangent = numpy.array([-self.normal[1], self.normal[0]])

        self.paths = [LinePath(v, tangent)]

    def dist(self, x):
        """
        Compute the distribution.

        Args:
            self: (todo): write your description
            x: (array): write your description
        """
        assert x.shape[0] == 2
        return self.alpha - numpy.dot(self.normal, x)

    def boundary_step(self, x):
        """
        Evaluates the boundary function.

        Args:
            self: (todo): write your description
            x: (todo): write your description
        """
        beta = self.alpha - numpy.dot(self.normal, x) / numpy.dot(
            self.normal, self.normal
        )
        return x + numpy.multiply.outer(self.normal, beta)
