import numpy

from ..helpers import multi_newton
from .geometry import Geometry


class Ellipse(Geometry):
    def __init__(self, x0, a, b):
        super().__init__()
        self.x0 = x0
        self.a = a
        self.b = b
        self.bounding_box = [x0[0] - a, x0[0] + a, x0[1] - b, x0[1] + b]
        self.feature_points = numpy.array([])

    def dist(self, x):
        assert x.shape[0] == 2
        return (
            ((x[0] - self.x0[0]) / self.a) ** 2
            + ((x[1] - self.x0[1]) / self.b) ** 2
            - 1.0
        )

    def _boundary_step(self, x):
        ax = (x[0] - self.x0[0]) / self.a
        ay = (x[1] - self.x0[1]) / self.b

        alpha = ax ** 2 + ay ** 2 - 1.0
        jac = numpy.array([4 * alpha * ax / self.a, 4 * alpha * ay / self.b])

        dalpha_dx = 2 * ax / self.a
        dalpha_dy = 2 * ay / self.b
        hess = numpy.array(
            [
                [
                    4 * dalpha_dx * ax / self.a + 4 * alpha / self.a ** 2,
                    4 * dalpha_dy * ax / self.a,
                ],
                [
                    4 * dalpha_dx * ay / self.b,
                    4 * dalpha_dy * ay / self.b + 4 * alpha / self.b ** 2,
                ],
            ]
        )

        p = -numpy.linalg.solve(numpy.moveaxis(hess, -1, 0), jac.T)
        return x + p.T

    def boundary_step(self, x):
        return multi_newton(
            x.T, self.dist, self._boundary_step, 1.0e-10, max_num_steps=10
        ).T
