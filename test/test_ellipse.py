# -*- coding: utf-8 -*-
#
import dmsh

from helpers import assert_norm_equality, save


def test_ellipse(show=True):
    geo = dmsh.Ellipse([0.0, 0.0], 2.0, 1.0)
    X, cells = dmsh.generate(geo, 0.2, show=show)

    # ref_norms = [1.9273675392415075e+01, 4.7336334768604962e+00, 2.0]
    # assert_norm_equality(
    #     X.flatten(), ref_norms, 1.0e-12
    # )
    return X, cells


if __name__ == "__main__":
    X, cells = test_ellipse(show=True)
    save("ellipse.png", X, cells)
