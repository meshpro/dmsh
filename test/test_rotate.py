# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality


def test(h0=0.9, show=True):
    geo = dmsh.Rotation(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), 0.1 * numpy.pi)
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells, [[1, 5, 2], [0, 1, 2], [3, 1, 4], [3, 5, 1], [6, 3, 4], [5, 3, 6]]
    )

    assert_norm_equality(
        X.flatten(),
        [1.3905252276781415e+01, 3.0919890978059588e+00, 9.9320349979775846e-01],
        1.0e-12,
    )
    return


if __name__ == "__main__":
    test(h0=0.1)
