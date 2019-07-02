import numpy

import dmsh
from helpers import assert_norm_equality, save


def test_union(show=False):
    angles = numpy.pi * numpy.array([3.0 / 6.0, 7.0 / 6.0, 11.0 / 6.0])
    geo = dmsh.Union(
        [
            dmsh.Circle([numpy.cos(angles[0]), numpy.sin(angles[0])], 1.0),
            dmsh.Circle([numpy.cos(angles[1]), numpy.sin(angles[1])], 1.0),
            dmsh.Circle([numpy.cos(angles[2]), numpy.sin(angles[2])], 1.0),
        ]
    )
    X, cells = dmsh.generate(geo, 0.2, show=show, tol=1.0e-10)

    ref_norms = [4.1390554922002769e02, 2.1440246410944471e01, 1.9947113226010518e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-12)
    return X, cells


if __name__ == "__main__":
    X, cells = test_union(show=False)
    save("union_three_circles.png", X, cells)
