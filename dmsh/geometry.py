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

    def boundary_step(self, x):
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
        return p


class Circle(Ellipse):
    def __init__(self, x0, r):
        super(Circle, self).__init__(x0, r, r)
        return


# class Rectangle(object):
#     def __init__(x0, x1, y0, y1):
#         return
#
#     def isinside(self, x):
#         assert x.shape[0] == 2
#         return (
#             ((x[0] - self.x0[0]) / self.a) ** 2
#             + ((x[1] - self.x0[1]) / self.b) ** 2
#             - 1.0
#         )
