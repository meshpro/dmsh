# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_union(h0=0.7, show=True):
    angles = numpy.pi * numpy.array([3.0 / 6.0, 7.0 / 6.0, 11.0 / 6.0])
    geo = dmsh.geometry.Intersection(
        [
            dmsh.geometry.Circle([numpy.cos(angles[0]), numpy.sin(angles[0])], 1.5),
            dmsh.geometry.Circle([numpy.cos(angles[1]), numpy.sin(angles[1])], 1.5),
            dmsh.geometry.Circle([numpy.cos(angles[2]), numpy.sin(angles[2])], 1.5),
        ]
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    print(cells)
    assert numpy.array_equal(
        cells,
        [[5, 6, 4], [2, 6, 1], [6, 2, 4], [6, 0, 1], [5, 7, 6], [0, 7, 3], [7, 0, 6]],
    )
    tol = 1.0e-12

    ref_norm1 = 5.398493299171385
    assert abs(numpy.linalg.norm(X.flatten(), ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 1.6297882283557934
    assert abs(numpy.linalg.norm(X.flatten(), ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 0.7247448713853791
    assert (
        abs(numpy.linalg.norm(X.flatten(), ord=numpy.inf) - ref_norm_inf)
        < tol * ref_norm_inf
    )
    return


if __name__ == "__main__":
    test_union(h0=0.1)
