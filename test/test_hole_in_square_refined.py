# -*- coding: utf-8 -*-
#
import numpy
import dmsh


def test(show=False):
    r = dmsh.Rectangle(-1.0, +1.0, -1.0, +1.0)
    c = dmsh.Circle([0.0, 0.0], 0.3)
    geo = dmsh.Difference(r, c)
    X, cells = dmsh.generate(
        geo,
        lambda pts: numpy.abs(c.dist(pts)) / 3 + 0.05,
        show=show,
    )

    # tol = 1.0e-12
    # ref_norm1 = 135.0
    # X = X.flatten()
    # assert abs(numpy.linalg.norm(X, ord=1) - ref_norm1) < tol * ref_norm1
    # ref_norm2 = 22.81303143058649
    # assert abs(numpy.linalg.norm(X, ord=2) - ref_norm2) < tol * ref_norm2
    # ref_norm_inf = 5.0
    # assert abs(numpy.linalg.norm(X, ord=numpy.inf) - ref_norm_inf) < tol * ref_norm_inf
    return


if __name__ == "__main__":
    test(show=True)
