from helpers import assert_norm_equality, save

import dmsh


def test(show=False):
    """
    Generate the covariance.

    Args:
        show: (bool): write your description
    """
    geo = dmsh.Scaling(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), 2.0)
    X, cells = dmsh.generate(geo, 0.1, show=show, tol=1.0e-5)

    ref_norms = [7.7120755551889915e03, 1.2509255305705049e02, 4.0000000000000000e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


if __name__ == "__main__":
    X, cells = test(show=False)
    save("scaling.png", X, cells)
