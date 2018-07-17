# -*- coding: utf-8 -*-
#
import numpy

from .helpers import multi_newton


class Union(object):
    def __init__(self, geometries):
        self.geometries = geometries
        self.bounding_box = [
            numpy.min([geo.bounding_box[0] for geo in geometries]),
            numpy.max([geo.bounding_box[1] for geo in geometries]),
            numpy.min([geo.bounding_box[2] for geo in geometries]),
            numpy.max([geo.bounding_box[3] for geo in geometries]),
        ]
        return

    def plot(self, color="b"):
        for geo in self.geometries:
            geo.plot()
        return

    def isinside(self, x):
        return numpy.min([geo.isinside(x) for geo in self.geometries], axis=0)

    def boundary_step(self, x):
        # step for the is_inside with the smallest value
        alpha = numpy.array([geo.isinside(x) for geo in self.geometries])
        alpha[alpha < 0] = numpy.inf
        idx = numpy.argmin(alpha, axis=0)
        for k, geo in enumerate(self.geometries):
            if numpy.any(idx == k):
                x[:, idx == k] = geo.boundary_step(x[:, idx == k])
        return x


class Intersection(object):
    def __init__(self, geometries):
        self.geometries = geometries
        self.bounding_box = [
            numpy.max([geo.bounding_box[0] for geo in geometries]),
            numpy.min([geo.bounding_box[1] for geo in geometries]),
            numpy.max([geo.bounding_box[2] for geo in geometries]),
            numpy.min([geo.bounding_box[3] for geo in geometries]),
        ]
        return

    def plot(self, color="b"):
        for geo in self.geometries:
            geo.plot()
        return

    def isinside(self, x):
        return numpy.max([geo.isinside(x) for geo in self.geometries], axis=0)

    def boundary_step(self, x, tol=1.0e-12):
        # step for the is_inside with the smallest value
        alpha = numpy.array([geo.isinside(x) for geo in self.geometries])
        while numpy.any(alpha > tol):
            # Only consider the nodes which are truly outside of the domain
            has_pos = numpy.any(alpha > tol, axis=0)
            x_pos = x[:, has_pos]
            alpha_pos = alpha[:, has_pos]

            alpha_pos[alpha_pos < tol] = numpy.inf
            idx = numpy.argmin(alpha_pos, axis=0)
            for k, geo in enumerate(self.geometries):
                if numpy.any(idx == k):
                    x_pos[:, idx == k] = geo.boundary_step(x_pos[:, idx == k])

            x[:, has_pos] = x_pos
            alpha = numpy.array([geo.isinside(x) for geo in self.geometries])
        return x


class Ellipse(object):
    def __init__(self, x0, a, b):
        self.x0 = x0
        self.a = a
        self.b = b
        self.bounding_box = [x0[0] - a, x0[0] + a, x0[1] - b, x0[1] + b]
        return

    def plot(self, color="b"):
        import matplotlib.pyplot as plt

        t = numpy.linspace(0.0, 2 * numpy.pi, 100)
        plt.plot(
            self.x0[0] + self.a * numpy.cos(t),
            self.x0[1] + self.b * numpy.sin(t),
            "-",
            color=color,
        )
        return

    def isinside(self, x):
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
            x.T, self.isinside, self._boundary_step, 1.0e-10, max_num_steps=10
        ).T


class Circle(object):
    def __init__(self, x0, r):
        self.x0 = x0
        self.r = r
        self.bounding_box = [x0[0] - r, x0[0] + r, x0[1] - r, x0[1] + r]
        return

    def plot(self, color="b"):
        import matplotlib.pyplot as plt

        t = numpy.linspace(0.0, 2 * numpy.pi, 100)
        plt.plot(
            self.x0[0] + self.r * numpy.cos(t),
            self.x0[1] + self.r * numpy.sin(t),
            "-",
            color=color,
        )
        return

    def isinside(self, x):
        assert x.shape[0] == 2
        return (
            ((x[0] - self.x0[0]) / self.r) ** 2
            + ((x[1] - self.x0[1]) / self.r) ** 2
            - 1.0
        )

    def boundary_step(self, x):
        # simply project onto the circle
        x = (x.T - self.x0).T
        r = numpy.sqrt(numpy.einsum("ij,ij->j", x, x))
        return ((x / r * self.r).T + self.x0).T


class Rectangle(object):
    def __init__(self, x0, x1, y0, y1):
        self.x0 = x0
        self.x1 = x1
        self.y0 = y0
        self.y1 = y1
        self.bounding_box = [x0, x1, y0, y1]
        return

    def plot(self, color="b"):
        import matplotlib.pyplot as plt

        plt.plot(
            [self.x0, self.x1, self.x1, self.x0, self.x0],
            [self.y0, self.y0, self.y1, self.y1, self.y0],
            "-",
            color=color,
        )
        return

    def isinside(self, x):
        assert x.shape[0] == 2
        return numpy.max(
            numpy.array(
                [self.x0 - x[0], x[0] - self.x1, self.y0 - x[1], x[1] - self.y1]
            ),
            axis=0,
        )

    def boundary_step(self, x):
        x[0] = numpy.maximum(x[0], numpy.full(x[0].shape, self.x0))
        x[0] = numpy.minimum(x[0], numpy.full(x[0].shape, self.x1))
        x[1] = numpy.maximum(x[1], numpy.full(x[1].shape, self.y0))
        x[1] = numpy.minimum(x[1], numpy.full(x[1].shape, self.y1))
        return x
