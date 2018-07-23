# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test_pacman(h0=0.6, show=True):
    geo = dmsh.Difference(
        dmsh.Circle([0.0, 0.0], 1.0),
        dmsh.Polygon([[0.0, 0.0], [1.5, 0.4], [1.5, -0.4]]),
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells, [[3, 0, 5], [0, 6, 5], [6, 0, 7], [0, 2, 7], [0, 4, 1], [3, 4, 0]]
    ), cells

    assert_norm_equality(
        X.flatten(), [8.7974713775133200e+00, 2.6457513110645903e+00, 1.0], 1.0e-12
    )
    return


if __name__ == "__main__":
    test_pacman(h0=0.2)
