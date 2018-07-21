# -*- coding: utf-8 -*-
#
import numpy

import dmsh


def test(h0=0.9, show=True):
    geo = dmsh.Rotation(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), 0.1 * numpy.pi)
    X, cells = dmsh.generate(geo, h0, show=show)

    assert numpy.array_equal(
        cells, [[1, 5, 2], [0, 1, 2], [3, 1, 4], [3, 5, 1], [6, 3, 4], [5, 3, 6]]
    )

    tol = 1.0e-12
    X = X.flatten()

    ref_norm1 = 13.462995177487391
    assert abs(numpy.linalg.norm(X, ord=1) - ref_norm1) < tol * ref_norm1
    ref_norm2 = 4.176326940419989
    assert abs(numpy.linalg.norm(X, ord=2) - ref_norm2) < tol * ref_norm2
    ref_norm_inf = 2.2111300269652543
    assert abs(numpy.linalg.norm(X, ord=numpy.inf) - ref_norm_inf) < tol * ref_norm_inf
    return


if __name__ == "__main__":
    test(h0=0.1)
