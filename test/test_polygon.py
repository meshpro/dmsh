from helpers import assert_norm_equality, save

import dmsh


def test(show=False):
    geo = dmsh.Polygon(
        [
            [0.0, 0.0],
            [1.1, 0.0],
            [1.2, 0.5],
            [0.7, 0.6],
            [2.0, 1.0],
            [1.0, 2.0],
            [0.5, 1.5],
        ]
    )
    # geo.show()
    X, cells = dmsh.generate(geo, 0.1, show=show)

    ref_norms = [4.1454432512302594e02, 2.1854133564894923e01, 2.0000000000000000e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-5)
    return X, cells


if __name__ == "__main__":
    X, cells = test(show=False)
    save("polygon.svg", X, cells)
