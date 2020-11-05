import numpy
from helpers import assert_norm_equality, save

import dmsh


def test_halfspace(show=False):
    """
    Generate x y - axis.

    Args:
        show: (bool): write your description
    """
    geo = dmsh.Intersection(
        [
            dmsh.HalfSpace(numpy.sqrt(0.5) * numpy.array([1.0, 1.0]), 0.0),
            dmsh.Circle([0.0, 0.0], 1.0),
        ]
    )
    X, cells = dmsh.generate(geo, 0.1, show=show)

    ref_norms = [1.6445927927312613e02, 1.0032815150498680e01, 9.9962808330128095e-01]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-7)
    return X, cells


if __name__ == "__main__":
    X, cells = test_halfspace(show=False)
    save("halfspace.png", X, cells)
