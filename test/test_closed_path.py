import numpy

from dmsh.geometry import pypathlib


def test_show():
    """
    Show pypath.

    Args:
    """
    path = pypathlib.ClosedPath([[0.0, 0.0], [1.0, 0.0], [1.1, 1.1], [0.1, 1.0]])
    path.show()


def test_convex():
    """
    Test if the given path on the given path.

    Args:
    """
    path = pypathlib.ClosedPath([[0.0, 0.0], [1.0, 0.0], [1.1, 1.1], [0.1, 1.0]])

    ref = 1.045
    assert abs(path.area - ref) < 1.0e-12 * ref
    assert path.positive_orientation
    assert all(path.is_convex_node)


def test_orientation():
    """
    Computes the orientation of the current node.

    Args:
    """
    path = pypathlib.ClosedPath([[0.1, 1.0], [1.1, 1.1], [1.0, 0.0], [0.0, 0.0]])

    ref = 1.045
    assert abs(path.area - ref) < 1.0e-12 * ref
    assert not path.positive_orientation
    assert all(path.is_convex_node)


def test_concave():
    """
    Concave isconcave.

    Args:
    """
    path = pypathlib.ClosedPath(
        [[0.0, 0.0], [1.0, 0.0], [0.9, 0.5], [1.1, 1.1], [0.1, 1.0]]
    )

    ref = 0.965
    assert abs(path.area - ref) < 1.0e-12 * ref
    assert path.positive_orientation
    assert numpy.array_equal(path.is_convex_node, [True, True, False, True, True])


def test_concave_counterclock():
    """
    Concave counter isconcave.

    Args:
    """
    path = pypathlib.ClosedPath(
        [[0.1, 1.0], [1.1, 1.1], [0.9, 0.5], [1.0, 0.0], [0.0, 0.0]]
    )

    ref = 0.965
    assert abs(path.area - ref) < 1.0e-12 * ref
    assert not path.positive_orientation
    assert numpy.array_equal(path.is_convex_node, [True, True, False, True, True])


def test_squared_distance():
    """
    Calculate distance between two pypath.

    Args:
    """
    path = pypathlib.ClosedPath(
        [[0.0, 0.0], [1.0, 0.0], [0.9, 0.5], [1.0, 1.0], [0.0, 1.0]]
    )

    dist = path.squared_distance(
        [[0.2, 0.1], [0.5, 0.5], [1.0, 0.5], [0.0, 1.1], [-0.1, 1.1], [1.0, 1.0]]
    )
    ref = numpy.array([0.01, 0.16, 1.0 / 104.0, 0.01, 0.02, 0.0])
    assert numpy.all(numpy.abs(dist - ref) < 1.0e-12)


def test_distance():
    """
    Calculate distance between two points.

    Args:
    """
    path = pypathlib.ClosedPath(
        [[0.0, 0.0], [1.0, 0.0], [0.9, 0.5], [1.0, 1.0], [0.0, 1.0]]
    )

    dist = path.distance(
        [[0.2, 0.1], [0.5, 0.5], [1.0, 0.5], [0.0, 1.1], [-0.1, 1.1], [1.0, 1.0]]
    )
    ref = numpy.array([0.1, 0.4, numpy.sqrt(1.0 / 104.0), 0.1, numpy.sqrt(2) / 10, 0.0])
    assert numpy.all(numpy.abs(dist - ref) < 1.0e-12)


def test_signed_distance():
    """
    Compute the distance between two points.

    Args:
    """
    path = pypathlib.ClosedPath(
        [[0.0, 0.0], [1.0, 0.0], [0.9, 0.5], [1.0, 1.0], [0.0, 1.0]]
    )

    dist = path.signed_distance(
        [[0.2, 0.1], [0.5, 0.5], [1.0, 0.5], [0.0, 1.1], [-0.1, 1.1], [1.0, 1.0]]
    )
    print(dist)
    ref = numpy.array(
        [-0.1, -0.4, numpy.sqrt(1.0 / 104.0), 0.1, numpy.sqrt(2) / 10, 0.0]
    )
    assert numpy.all(numpy.abs(dist - ref) < 1.0e-12)


def test_inside():
    """
    Test if the pypath is in - place.

    Args:
    """
    path = pypathlib.ClosedPath(
        [[0.0, 0.0], [1.0, 0.0], [0.9, 0.5], [1.0, 1.0], [0.0, 1.0]]
    )

    contains_points = path.contains_points(
        [[0.2, 0.1], [0.5, 0.5], [1.0, 0.5], [0.0, 1.1], [-0.1, 1.1], [1.0, 1.0]]
    )
    assert numpy.array_equal(contains_points, [True, True, False, False, False, True])


def test_closest_points():
    """
    Test if the closest point on pypath.

    Args:
    """
    path = pypathlib.ClosedPath(
        [[0.0, 0.0], [1.0, 0.0], [0.9, 0.5], [1.0, 1.0], [0.0, 1.0]]
    )

    closest_points = path.closest_points(
        [[0.2, 0.1], [0.5, 0.5], [1.0, 0.5], [0.0, 1.1], [-0.1, 1.1], [1.0, 1.0]]
    )

    ref = numpy.array(
        [
            [0.2, 0.0],
            [0.9, 0.5],
            [9.0384615384615385e-01, 5.1923076923076927e-01],
            [0.0, 1.0],
            [0.0, 1.0],
            [1.0, 1.0],
        ]
    )
    assert numpy.all(numpy.abs(closest_points - ref) < 1.0e-12)


def test_signed_squared_distance():
    """
    Calculate the distance between two distance between two files.

    Args:
    """
    path = pypathlib.ClosedPath(
        [[0.0, 0.0], [1.0, 0.0], [0.9, 0.5], [1.0, 1.0], [0.0, 1.0]]
    )

    dist = path.signed_squared_distance(
        [[0.2, 0.1], [0.5, 0.5], [1.0, 0.5], [0.0, 1.1], [-0.1, 1.1], [1.0, 1.0]]
    )
    ref = numpy.array([-0.01, -0.16, 1.0 / 104.0, 0.01, 0.02, 0.0])
    assert numpy.all(numpy.abs(dist - ref) < 1.0e-12)


def test_sharp_angle():
    """
    Test if angle angle between two angle.

    Args:
    """
    path = pypathlib.ClosedPath(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.45],
            [0.6, 0.5],
            [1.0, 0.55],
            [1.0, 1.0],
            [0.0, 1.0],
        ]
    )

    contains_points = path.contains_points([[0.5, 0.4], [0.5, 0.6]])
    assert numpy.all(contains_points)

    dist = path.signed_squared_distance([[0.5, 0.4], [0.5, 0.6]])
    ref = numpy.array([-0.02, -0.02])
    assert numpy.all(numpy.abs(dist - ref) < 1.0e-12)


# def test_two_points():
#     path = pypathlib.ClosedPath([[-0.5, 1.0], [+0.5, 1.0]])
#     contains_points = path.contains_points([[0.0, 0.0], [0.0, 2.0]])
#     assert numpy.array_equal(contains_points, [False, False])


if __name__ == "__main__":
    test_sharp_angle()
