# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test_intersection(h0=0.5, show=True):
    geo = dmsh.Intersection(
        [dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0)]
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells, [[0, 2, 3], [2, 0, 4], [5, 1, 3], [1, 5, 4], [2, 5, 3], [5, 2, 4]]
    ), cells

    assert_norm_equality(
        X.flatten(),
        [3.3225196794944698e+00, 1.4745598476401574e+00, 8.6602540377406145e-01],
        1.0e-12,
    )
    return


if __name__ == "__main__":
    test_intersection(0.1)
