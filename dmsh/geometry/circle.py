import numpy

from .geometry import Geometry


class CirclePath:
    def __init__(self, x0, r):
        self.x0 = x0
        self.r = r

    def p(self, t):
        v = numpy.array([numpy.cos(2 * numpy.pi * t), numpy.sin(2 * numpy.pi * t)])
        return ((self.r * v).T + self.x0).T

    def dp_dt(self, t):
        return (
            self.r
            * 2
            * numpy.pi
            * numpy.array([-numpy.sin(2 * numpy.pi * t), numpy.cos(2 * numpy.pi * t)])
        )


class Circle(Geometry):
    def __init__(self, x0, r):
        super().__init__()
        self.x0 = x0
        self.r = r
        self.bounding_box = [x0[0] - r, x0[0] + r, x0[1] - r, x0[1] + r]
        self.paths = [CirclePath(x0, r)]
        self.feature_points = numpy.array([[], []]).T

    def dist(self, x):
        assert x.shape[0] == 2
        y = (x.T - self.x0).T
        return numpy.sqrt(numpy.einsum("i...,i...->...", y, y)) - self.r

    def boundary_step(self, x):
        # simply project onto the circle
        y = (x.T - self.x0).T
        r = numpy.sqrt(numpy.einsum("ij,ij->j", y, y))
        return ((y / r * self.r).T + self.x0).T

    def plot(self, level_set=True):
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
