import numpy
from helpers import assert_norm_equality, save

import dmsh


def test_union(show=False):
    """
    Test the union of the union

    Args:
        show: (bool): write your description
    """
    angles = numpy.pi * numpy.array([3.0 / 6.0, 7.0 / 6.0, 11.0 / 6.0])
    geo = dmsh.Intersection(
        [
            dmsh.Circle([numpy.cos(angles[0]), numpy.sin(angles[0])], 1.5),
            dmsh.Circle([numpy.cos(angles[1]), numpy.sin(angles[1])], 1.5),
            dmsh.Circle([numpy.cos(angles[2]), numpy.sin(angles[2])], 1.5),
        ]
    )
    X, cells = dmsh.generate(geo, 0.1, show=show, tol=1.0e-10)

    ref_norms = [6.9568161823685358e01, 5.1355079813279527e00, 7.2474487138537913e-01]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


if __name__ == "__main__":
    X, cells = test_union(show=False)
    save("intersection_three_circles.png", X, cells)
