import numpy
import perfplot
import pytest
from matplotlib import path

from dmsh.geometry import pypathlib


@pytest.mark.skip(reason="fails on gh-actions for some reason")
def test_speed(n=3):
    """
    Test if n times.

    Args:
        n: (str): write your description
    """
    path_pts = [[0, 0], [0, 1], [1, 1], [1, 0]]
    path0 = path.Path(path_pts)
    path1 = pypathlib.ClosedPath(path_pts)

    def _mpl_path(pts):
        """
        Returns a list of a list of points.

        Args:
            pts: (todo): write your description
        """
        return path0.contains_points(pts)

    def _pypathlib_contains_points(pts):
        """
        Return true if a list of points contains a list of the given points.

        Args:
            pts: (todo): write your description
        """
        return path1.contains_points(pts)

    numpy.random.seed(0)

    perfplot.show(
        setup=lambda n: numpy.random.rand(n, 2),
        kernels=[_mpl_path, _pypathlib_contains_points],
        n_range=[2 ** k for k in range(n)],
        labels=["matplotlib.path.contains_points", "pypathlib.contains_points"],
        logx=True,
        logy=True,
        xlabel="num points",
    )


def benchmark():
    """
    Benchmark a set of the image.

    Args:
    """
    path_pts = [[0, 0], [0, 1], [1, 1], [1, 0]]
    path1 = pypathlib.ClosedPath(path_pts)
    pts = numpy.random.rand(5000000, 2)
    path1.contains_points(pts)


if __name__ == "__main__":
    # test_speed(20)
    benchmark()
