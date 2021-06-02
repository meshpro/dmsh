import numpy as np

from .geometry import Geometry


class LinePath:
    def __init__(self, v, tangent):
        super().__init__()
        self.v = v
        self.tangent = tangent

    def p(self, t):
        """This parametrization of the line is (inf, inf) for t=0 and t=1."""
        # Don't warn on division by 0
        with np.errstate(divide="ignore"):
            out = (
                np.multiply.outer(self.tangent, (2 * t - 1) / t / (1 - t)).T + self.v
            ).T
        return out

    def dp_dt(self, t):
        with np.errstate(divide="ignore"):
            dt = 1 / t ** 2 + 1 / (1 - t) ** 2
        return np.multiply.outer(self.tangent, dt)


class HalfSpace(Geometry):
    def __init__(self, normal, alpha=0.0):
        super().__init__()
        self.normal = normal
        self.alpha = alpha

        self.bounding_box = [-np.inf, +np.inf, -np.inf, +np.inf]
        self.feature_points = np.array([])

        # One point on the line:
        v = self.normal / np.dot(self.normal, self.normal) * self.alpha
        tangent = np.array([-self.normal[1], self.normal[0]])

        self.paths = [LinePath(v, tangent)]

    def dist(self, x):
        assert x.shape[0] == 2
        out = self.alpha - np.dot(self.normal, x.reshape(x.shape[0], -1))
        return out.reshape(x.shape[1:])

    def boundary_step(self, x):
        beta = self.alpha - np.dot(self.normal, x) / np.dot(self.normal, self.normal)
        return x + np.multiply.outer(self.normal, beta)
