# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test(h0=0.7, show=True):
    geo = dmsh.Rectangle(0.0, 1.0, 0.0, 1.0)

    # p0 = dmsh.Path([[0.0, 0.0]])
    p1 = dmsh.Path([[0.4, 0.6], [0.6, 0.4]])

    def edge_size(x):
        return 0.03 + h0 * p1.dist(x)

    numpy.random.seed(0)
    X, cells = dmsh.generate(geo, edge_size, show=show, tol=1.0e-4)

    assert_norm_equality(
        X.flatten(), [6.4015362904210633e+01, 6.1707935698898639e+00, 1.0], 1.0e-12
    )
    return


if __name__ == "__main__":
    test(0.7)
