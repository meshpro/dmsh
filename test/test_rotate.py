# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test(h0=0.9, show=True):
    geo = dmsh.Rotation(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), 0.1 * numpy.pi)
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells, [[6, 3, 4], [3, 5, 2], [5, 3, 6], [0, 3, 2], [3, 1, 4], [0, 1, 3]]
    ), cells

    assert_norm_equality(
        X.flatten(),
        [1.2993750765965753e+01, 4.0926763862103552e+00, 2.2111300269652543e+00],
        1.0e-12,
    )
    return


if __name__ == "__main__":
    test(h0=0.1)
