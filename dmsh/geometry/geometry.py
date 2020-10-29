import numpy


class Geometry:
    def __init__(self):
        return

    def _get_xyz(self, nx=101, ny=101):
        x0, x1, y0, y1 = self.bounding_box
        w = x1 - x0
        h = x1 - x0
        x = numpy.linspace(x0 - w * 0.1, x1 + w * 0.1, nx)
        y = numpy.linspace(y0 - h * 0.1, y1 + h * 0.1, ny)
        X, Y = numpy.meshgrid(x, y)
        Z = self.dist(numpy.array([X, Y]))
        return X, Y, Z

    def _plot_level_set(self):
        import matplotlib.pyplot as plt

        X, Y, Z = self._get_xyz()
        alpha = numpy.max(numpy.abs(Z))
        cf = plt.contourf(
            X, Y, Z, levels=20, cmap=plt.cm.coolwarm, vmin=-alpha, vmax=alpha
        )
        plt.colorbar(cf)

    def plot(self, level_set=True):
        import matplotlib.pyplot as plt

        X, Y, Z = self._get_xyz()

        if level_set:
            alpha = numpy.max(numpy.abs(Z))
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
