# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test(h0=0.9, show=True):
    geo = dmsh.Stretch(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), [1.0, 1.0])
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [
            [8, 4, 5],
            [1, 4, 0],
            [4, 1, 5],
            [7, 6, 3],
            [9, 8, 5],
            [6, 9, 5],
            [6, 2, 3],
            [1, 2, 5],
            [2, 6, 5],
            [10, 9, 6],
            [10, 7, 11],
            [7, 10, 6],
        ],
    )

    assert_norm_equality(
        X.flatten(),
        [1.3905252276781415e+01, 3.0919890978059588e+00, 9.9320349979775846e-01],
        1.0e-12,
    )
    return


if __name__ == "__main__":
    test(h0=0.1)
