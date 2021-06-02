import numpy as np
from helpers import assert_norm_equality, save

import dmsh


def test_halfspace(show=False):
    geo = dmsh.Intersection(
        [
            dmsh.HalfSpace(np.sqrt(0.5) * np.array([1.0, 1.0])),
            dmsh.Circle([0.0, 0.0], 1.0),
        ]
    )
    X, cells = dmsh.generate(geo, 0.1, show=show, max_steps=100)

    ref_norms = [1.6399670188761661e02, 1.0011048291798387e01, 9.9959986881486440e-01]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-6)
    return X, cells


if __name__ == "__main__":
    X, cells = test_halfspace(show=True)
    save("halfspace.png", X, cells)
