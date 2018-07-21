# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test_circle(h0=0.7, show=True):
    geo = dmsh.Circle([0.0, 0.0], 1.0)
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells, [[3, 5, 2], [0, 3, 2], [6, 3, 4], [3, 6, 5], [3, 1, 4], [0, 1, 3]]
    )

    assert_norm_equality(
        X.flatten(),
        [6.2697392291785894e+00, 2.0575365561277703e+00, 8.3998578160606641e-01],
        1.0e-12,
    )
    return


if __name__ == "__main__":
    test_circle(0.3)
