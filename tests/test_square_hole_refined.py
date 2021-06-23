import numpy as np
from helpers import assert_norm_equality, save

import dmsh


def test(show=False):
    r = dmsh.Rectangle(-1.0, +1.0, -1.0, +1.0)
    c = dmsh.Circle([0.0, 0.0], 0.3)
    geo = dmsh.Difference(r, c)

    X, cells = dmsh.generate(
        geo,
        lambda pts: np.abs(c.dist(pts)) / 5 + 0.05,
        show=show,
        tol=1.0e-10,
        max_steps=100,
    )

    ref_norms = [2.3686099753024831e02, 1.1750558136202198e01, 1.0000000000000000e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-2)
    return X, cells


if __name__ == "__main__":
    X, cells = test(show=True)
    save("square_hole_refined.png", X, cells)
