from helpers import assert_norm_equality, save

import dmsh


def test_intersection(show=False):
    """
    Generate the intersection of the geometries.

    Args:
        show: (bool): write your description
    """
    geo = dmsh.Intersection(
        [dmsh.Circle([0.0, -0.5], 1.0), dmsh.Circle([0.0, +0.5], 1.0)]
    )
    X, cells = dmsh.generate(geo, 0.1, show=show, tol=1.0e-10)

    geo.plot()

    ref_norms = [8.6619344595913475e01, 6.1599895121114274e00, 8.6602540378466342e-01]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-10)
    return X, cells


if __name__ == "__main__":
    X, cells = test_intersection(show=False)
    save("intersection.png", X, cells)
