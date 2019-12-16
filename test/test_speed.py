import numpy
from matplotlib import path

import perfplot
from dmsh.geometry import pypathlib


def test_speed(n=3):
    path_pts = [[0, 0], [0, 1], [1, 1], [1, 0]]
    path0 = path.Path(path_pts)
    path1 = pypathlib.ClosedPath(path_pts)

    def _mpl_path(pts):
        return path0.contains_points(pts)

    def _pypathlib_contains_points(pts):
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
    return


def benchmark():
    path_pts = [[0, 0], [0, 1], [1, 1], [1, 0]]
    path1 = pypathlib.ClosedPath(path_pts)
    pts = numpy.random.rand(5000000, 2)
    path1.contains_points(pts)
    return


if __name__ == "__main__":
    # test_speed(20)
    benchmark()
