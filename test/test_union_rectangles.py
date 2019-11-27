import dmsh
from helpers import assert_norm_equality, save


def test_union(show=False):
    geo = dmsh.Union(
        [dmsh.Rectangle(-1.0, +0.5, -1.0, +0.5), dmsh.Rectangle(-0.5, +1.0, -0.5, +1.0)]
    )
    X, cells = dmsh.generate(geo, 0.15, show=show, tol=1.0e-5)

    ref_norms = [1.7869153426373094e02, 1.1117125741661049e01, 1.0]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-12)
    return X, cells


if __name__ == "__main__":
    X, cells = test_union(show=True)
    save("union_rectangles.png", X, cells)
