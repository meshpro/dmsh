# -*- coding: utf-8 -*-
#
import dmsh

from helpers import assert_norm_equality, save


def test_circle(show=True):
    geo = dmsh.Circle([0.0, 0.0], 1.0)
    X, cells = dmsh.generate(geo, 0.1, show=show)

    ref_norms = [3.2795193920779542e+02, 1.4263721858241993e+01, 1.0000000000000000e+00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-12)
    return X, cells


if __name__ == "__main__":
    X, cells = test_circle(show=False)
    save("circle.png", X, cells)
