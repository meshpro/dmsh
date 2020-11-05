import numpy
from helpers import assert_norm_equality, save

import dmsh


def test(show=False):
    """
    Generate x and y and dms.

    Args:
        show: (bool): write your description
    """
    geo = dmsh.Rotation(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), 0.1 * numpy.pi)
    X, cells = dmsh.generate(geo, 0.1, show=show, tol=1.0e-10)

    ref_norms = [9.5352192763006087e02, 3.1344318120303605e01, 2.2111300269652543e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


if __name__ == "__main__":
    X, cells = test(show=False)
    save("rotation.png", X, cells)
