from helpers import assert_norm_equality, save

import dmsh


def test_pacman(show=False):
    """
    Generate test isochman test.

    Args:
        show: (bool): write your description
    """
    geo = dmsh.Difference(
        dmsh.Circle([0.0, 0.0], 1.0),
        dmsh.Polygon([[0.0, 0.0], [1.5, 0.4], [1.5, -0.4]]),
    )
    X, cells = dmsh.generate(geo, 0.1, show=show, tol=1.0e-10)

    ref_norms = [3.0305576561748495e02, 1.3611677164248055e01, 1.0]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


if __name__ == "__main__":
    X, cells = test_pacman(show=False)
    save("pacman.png", X, cells)
