import numpy


class Geometry:
    def __init__(self):
        """
        Initialize a function

        Args:
            self: (todo): write your description
        """
        return

    def _get_xyz(self, nx=101, ny=101):
        """
        Return x y z coordinates of the x y z coordinates.

        Args:
            self: (todo): write your description
            nx: (todo): write your description
            ny: (todo): write your description
        """
        x0, x1, y0, y1 = self.bounding_box
        w = x1 - x0
        h = x1 - x0
        x = numpy.linspace(x0 - w * 0.1, x1 + w * 0.1, nx)
        y = numpy.linspace(y0 - h * 0.1, y1 + h * 0.1, ny)
        X, Y = numpy.meshgrid(x, y)
        Z = self.dist(numpy.array([X, Y]))
        return X, Y, Z

    def _plot_level_set(self):
        """
        Plot the level of the contour.

        Args:
            self: (todo): write your description
        """
        import matplotlib.pyplot as plt

        X, Y, Z = self._get_xyz()
        alpha = numpy.max(numpy.abs(Z))
        cf = plt.contourf(
            X, Y, Z, levels=20, cmap=plt.cm.coolwarm, vmin=-alpha, vmax=alpha
        )
        plt.colorbar(cf)

    def plot(self, level_set=True):
        """
        Plot the 2d plot.

        Args:
            self: (todo): write your description
            level_set: (todo): write your description
        """
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
        """
        Displays the plot.

        Args:
            self: (todo): write your description
        """
        import matplotlib.pyplot as plt

        self.plot(*args, **kwargs)
        plt.show()
