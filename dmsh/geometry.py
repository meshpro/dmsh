# -*- coding: utf-8 -*-
#
import numpy


# class Union(object):
#     def __init__(self, geometries):
#         self.geometries = geometries
#         return
#
#     def isinside(self, x):
#         return


class Circle(object):
    def __init__(self, x0, r):
        self.x0 = x0
        self.r = r
        self.bounding_box = [x0[0] - r, x0[0] + r, x0[1] - r, x0[1] + r]
        return

    def plot(self, color="b"):
        import matplotlib.pyplot as plt

        t = numpy.linspace(0.0, 2 * numpy.pi, 100)
        plt.plot(self.r * numpy.cos(t), self.r * numpy.sin(t), "-", color=color)
        return

    def isinside(self, x):
        assert x.shape[0] == 2
        return (x[0] - self.x0[0]) ** 2 + (x[1] - self.x0[1]) ** 2 - self.r ** 2

    def jac2(self, x):
        assert x.shape[0] == 2
        # TODO self.x0
        return numpy.array(
            [
                4 * (x[0] ** 2 + x[1] ** 2 - self.r ** 2) * x[0],
                4 * (x[0] ** 2 + x[1] ** 2 - self.r ** 2) * x[1],
            ]
        )

    def hessian2(self, x):
        assert x.shape[0] == 2
        # TODO self.x0
        return numpy.array(
            [
                [
                    8 * x[0] ** 2 + 4 * (x[0] ** 2 + x[1] ** 2 - self.r ** 2),
                    8 * x[0] * x[1],
                ],
                [
                    8 * x[0] * x[1],
                    8 * x[1] ** 2 + 4 * (x[0] ** 2 + x[1] ** 2 - self.r ** 2),
                ],
            ]
        )


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
        plt.plot(self.a * numpy.cos(t), self.b * numpy.sin(t), "-", color=color)
        return

    def isinside(self, x):
        assert x.shape[0] == 2
        return (
            ((x[0] - self.x0[0]) / self.a) ** 2
            + ((x[1] - self.x0[1]) / self.b) ** 2
            - 1.0
        )

    def jac2(self, x):
        assert x.shape[0] == 2
        alpha = (
            ((x[0] - self.x0[0]) / self.a) ** 2
            + ((x[1] - self.x0[1]) / self.b) ** 2
            - 1.0
        )
        return numpy.array(
            [
                4 * alpha * (x[0] - self.x0[0]) / self.a ** 2,
                4 * alpha * (x[1] - self.x0[1]) / self.b ** 2,
            ]
        )

    def hessian2(self, x):
        assert x.shape[0] == 2
        alpha = (
            ((x[0] - self.x0[0]) / self.a) ** 2
            + ((x[1] - self.x0[1]) / self.b) ** 2
            - 1.0
        )
        dalpha_dx = 2 * (x[0] - self.x0[0]) / self.a ** 2
        dalpha_dy = 2 * (x[1] - self.x0[1]) / self.b ** 2
        return numpy.array(
            [
                [
                    4 * dalpha_dx * (x[0] - self.x0[0]) / self.a ** 2
                    + 4 * alpha / self.a ** 2,
                    4 * dalpha_dy * (x[0] - self.x0[0]) / self.a ** 2,
                ],
                [
                    4 * dalpha_dx * (x[1] - self.x0[1]) / self.b ** 2,
                    4 * dalpha_dy * (x[1] - self.x0[1]) / self.b ** 2
                    + 4 * alpha / self.b ** 2,
                ],
            ]
        )
