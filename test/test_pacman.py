# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test_pacman(h0=0.6, show=True):
    geo = dmsh.geometry.Difference(
        dmsh.geometry.Circle([0.0, 0.0], 1.0),
        dmsh.geometry.Polygon([[0.0, 0.0], [1.5, 0.4], [1.5, -0.4]]),
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [0, 7, 6],
            [7, 0, 8],
            [0, 9, 8],
            [3, 0, 6],
            [3, 4, 0],
            [0, 5, 1],
            [4, 5, 0],
            [2, 9, 0],
        ],
    )
    tol = 1.0e-12

    ref_norm1 = 11.287606084722736
    assert abs(numpy.linalg.norm(X.flatten(), ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 3.0
    assert abs(numpy.linalg.norm(X.flatten(), ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 1.0
    assert (
        abs(numpy.linalg.norm(X.flatten(), ord=numpy.inf) - ref_norm_inf)
        < tol * ref_norm_inf
    )
    return


if __name__ == "__main__":
    test_pacman(h0=0.2)
