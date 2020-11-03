from helpers import assert_norm_equality, save

import dmsh


def test_ellipse(show=False):
    geo = dmsh.Ellipse([0.0, 0.0], 2.0, 1.0)
    X, cells = dmsh.generate(geo, 0.2, show=show)

    geo.plot()

    ref_norms = [2.5238407704910199e02, 1.5704208202824834e01, 1.9999857285273639e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-2)
    return X, cells


if __name__ == "__main__":
    X, cells = test_ellipse(show=True)
    save("ellipse.png", X, cells)
