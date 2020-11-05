from helpers import assert_norm_equality, save

import dmsh


def test(show=True):
    """
    Generate x and y and dms.

    Args:
        show: (bool): write your description
    """
    geo = dmsh.Stretch(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), [1.0, 1.0])
    X, cells = dmsh.generate(geo, 0.3, show=show, tol=1.0e-3)

    ref_norms = [1.9391178579025609e02, 1.5890693098212086e01, 2.6213203435596428e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-2)
    return X, cells


if __name__ == "__main__":
    X, cells = test(show=False)
    save("stretch.png", X, cells)
