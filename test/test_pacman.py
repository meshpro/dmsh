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
    ), cells

    assert_norm_equality(
        X.flatten(),
        [1.3905252276781415e+01, 3.0919890978059588e+00, 9.9320349979775846e-01],
        1.0e-12,
    )
    return


if __name__ == "__main__":
    test_pacman(h0=0.2)
