import numpy

from dmsh.geometry import pypathlib


def test_squared_distance():
    """
    Calculate distance between two sources.

    Args:
    """
    path = pypathlib.Path([[0.0, 0.0], [1.0, 0.0], [0.9, 0.5], [1.0, 1.0], [0.0, 1.0]])

    dist = path.squared_distance(
        [[0.2, 0.1], [0.5, 0.5], [1.0, 0.5], [0.0, 1.1], [-0.1, 1.1], [1.0, 1.0]]
    )
    ref = numpy.array([0.01, 0.16, 1.0 / 104.0, 0.01, 0.02, 0.0])
    assert numpy.all(numpy.abs(dist - ref) < 1.0e-12)
    return


def test_one_point():
    """
    Finds the point is in - point.

    Args:
    """
    path = pypathlib.Path([[0.0, 0.0]])

    dist = path.squared_distance(
        [[0.2, 0.1], [0.5, 0.5], [1.0, 0.5], [0.0, 1.1], [-0.1, 1.1], [1.0, 1.0]]
    )
    ref = numpy.array([0.05, 0.5, 1.25, 1.21, 1.22, 2.0])
    assert numpy.all(numpy.abs(dist - ref) < 1.0e-12)
    return


if __name__ == "__main__":
    test_squared_distance()
