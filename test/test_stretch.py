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
            [9, 6, 10],
            [6, 9, 5],
            [1, 2, 5],
            [6, 2, 3],
            [2, 6, 5],
            [4, 1, 5],
            [1, 4, 0],
            [9, 8, 5],
            [8, 4, 5],
            [4, 8, 7],
        ],
    ), cells

    assert_norm_equality(
        X.flatten(),
        [2.4150442183128522e+01, 6.1207354608584614e+00, 2.6213203435596428e+00],
        1.0e-12,
    )
    return


if __name__ == "__main__":
    test(h0=0.1)
