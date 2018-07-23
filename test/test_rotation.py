# -*- coding: utf-8 -*-
#
import numpy

import dmsh

from helpers import assert_norm_equality, save


def test(show=True):
    geo = dmsh.Rotation(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), 0.1 * numpy.pi)
    X, cells = dmsh.generate(geo, 0.1, show=show, tol=1.0e-10)

    ref_norms = [9.5457720168192884e+02, 3.1356929329612782e+01, 2.2111300269652543e+00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-12)
    return X, cells


if __name__ == "__main__":
    X, cells = test(show=False)
    save("rotate.png", X, cells)
