# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test_union(h0=0.7, show=True):
    angles = numpy.pi * numpy.array([3.0 / 6.0, 7.0 / 6.0, 11.0 / 6.0])
    geo = dmsh.Intersection(
        [
            dmsh.Circle([numpy.cos(angles[0]), numpy.sin(angles[0])], 1.5),
            dmsh.Circle([numpy.cos(angles[1]), numpy.sin(angles[1])], 1.5),
            dmsh.Circle([numpy.cos(angles[2]), numpy.sin(angles[2])], 1.5),
        ]
    )
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells,
        [[2, 7, 1], [7, 0, 1], [0, 7, 3], [7, 5, 3], [2, 6, 7], [5, 6, 4], [6, 5, 7]],
    )

    assert_norm_equality(
        X.flatten(),
        [5.4553963936720926e+00, 1.6297882995923110e+00, 7.2474487138537913e-01],
        1.0e-12,
    )
    return


if __name__ == "__main__":
    test_union(h0=0.1)
