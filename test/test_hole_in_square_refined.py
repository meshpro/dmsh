# -*- coding: utf-8 -*-
#
import numpy
import dmsh

from helpers import assert_norm_equality


def test(h=0.5, show=False):
    r = dmsh.Rectangle(-1.0, +1.0, -1.0, +1.0)
    c = dmsh.Circle([0.0, 0.0], 0.3)
    geo = dmsh.Difference(r, c)

    numpy.random.seed(0)
    X, cells = dmsh.generate(geo, lambda pts: numpy.abs(c.dist(pts)) / 3 + h, show=show)

    assert_norm_equality(
        X.flatten(),
        [1.6643623876679150e+01, 3.7169077843699139e+00, 1.0000000000000000e+00],
        1.0e-12,
    )
    return


if __name__ == "__main__":
    test(0.05, show=True)
