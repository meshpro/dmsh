from helpers import assert_norm_equality, save

import dmsh


def test(show=False):
    geo = dmsh.Stretch(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), [1.0, 1.0])
    X, cells = dmsh.generate(geo, 0.3, show=show, tol=1.0e-3, max_steps=100)

    ref_norms = [1.9006907971528796e02, 1.5666202908904914e01, 2.6213203435596428e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-2)
    return X, cells


if __name__ == "__main__":
    X, cells = test(show=False)
    save("stretch.png", X, cells)
