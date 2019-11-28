import numpy


class Geometry:
    def __init__(self):
        return

    def plot(self, level_set=True):
        import matplotlib.pyplot as plt

        x0, x1, y0, y1 = self.bounding_box

        w = x1 - x0
        h = x1 - x0
        x = numpy.linspace(x0 - w * 0.1, x1 + w * 0.1, 101)
        y = numpy.linspace(y0 - h * 0.1, y1 + h * 0.1, 101)
        X, Y = numpy.meshgrid(x, y)

        Z = self.dist(numpy.array([X, Y]))

        if level_set:
            alpha = max([abs(numpy.min(Z)), abs(numpy.min(Z))])
            cf = plt.contourf(
                X, Y, Z, levels=20, cmap=plt.cm.coolwarm, vmin=-alpha, vmax=alpha
            )
            plt.colorbar(cf)

        # mark the 0-level (the domain boundary)
        plt.contour(X, Y, Z, levels=[0.0], colors="k")

        plt.gca().set_aspect("equal")

    def show(self, *args, **kwargs):
        import matplotlib.pyplot as plt

        self.plot(*args, **kwargs)
        plt.show()
