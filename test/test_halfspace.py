# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality, save


def test_halfspace(show=False):
    geo = dmsh.Intersection(
        [
            dmsh.HalfSpace(numpy.sqrt(0.5) * numpy.array([1.0, 1.0]), 0.0),
            dmsh.Circle([0.0, 0.0], 1.0),
        ]
    )
    X, cells = dmsh.generate(geo, 0.1, show=show)

    ref_norms = [1.6445971629723411e+02, 1.0032823867864321e+01, 9.9962000746451751e-01]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-12)
    return X, cells


if __name__ == "__main__":
    X, cells = test_halfspace(show=False)
    save("halfspace.png", X, cells)
