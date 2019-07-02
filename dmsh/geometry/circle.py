import numpy


class CirclePath():
    def __init__(self, x0, r):
        self.x0 = x0
        self.r = r
        return

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


class Circle():
    def __init__(self, x0, r):
        self.x0 = x0
        self.r = r
        self.bounding_box = [x0[0] - r, x0[0] + r, x0[1] - r, x0[1] + r]
        self.paths = [CirclePath(x0, r)]
        self.feature_points = numpy.array([[], []]).T
        return

    def plot(self, color="#1f77b4"):
        import matplotlib.pyplot as plt

        t = numpy.linspace(0.0, 2 * numpy.pi, 100)
        plt.plot(
            self.x0[0] + self.r * numpy.cos(t),
            self.x0[1] + self.r * numpy.sin(t),
            "-",
            color=color,
        )
        return

    def dist(self, x):
        assert x.shape[0] == 2
        y = (x.T - self.x0).T
        return numpy.sqrt(numpy.einsum("ij,ij->j", y, y)) - self.r

    def boundary_step(self, x):
        # simply project onto the circle
        y = (x.T - self.x0).T
        r = numpy.sqrt(numpy.einsum("ij,ij->j", y, y))
        return ((y / r * self.r).T + self.x0).T
