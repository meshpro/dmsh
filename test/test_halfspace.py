import numpy
from helpers import assert_norm_equality, save

import dmsh


def test_halfspace(show=False):
    geo = dmsh.Intersection(
        [
            dmsh.HalfSpace(numpy.sqrt(0.5) * numpy.array([1.0, 1.0]), 0.0),
            dmsh.Circle([0.0, 0.0], 1.0),
        ]
    )
    X, cells = dmsh.generate(geo, 0.1, show=show)

    ref_norms = [1.6391103448060974e+02, 1.0020943433490697e+01, 9.9999732753403403e-01]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-6)
    return X, cells


if __name__ == "__main__":
    X, cells = test_halfspace(show=True)
    save("halfspace.png", X, cells)
