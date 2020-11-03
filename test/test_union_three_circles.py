import numpy
from helpers import assert_norm_equality, save

import dmsh


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

    ref_norms = [4.0376529583046920e02, 2.1158222355702804e01, 1.9999334075152841e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


if __name__ == "__main__":
    X, cells = test_union(show=False)
    save("union_three_circles.png", X, cells)
