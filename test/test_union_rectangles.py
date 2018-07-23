# -*- coding: utf-8 -*-
#
import dmsh

from helpers import assert_norm_equality, save


def test_union(show=True):
    geo = dmsh.Union(
        [dmsh.Rectangle(-1.0, +0.5, -1.0, +0.5), dmsh.Rectangle(-0.5, +1.0, -0.5, +1.0)]
    )
    X, cells = dmsh.generate(geo, 0.15, show=show, tol=1.0e-10)

    ref_norms = [1.7868961429998612e+02, 1.1117047580567053e+01, 1.0]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-12)
    return X, cells


if __name__ == "__main__":
    X, cells = test_union(show=False)
    save("union_rectangles.png", X, cells)
