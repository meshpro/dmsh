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
    X, cells = dmsh.generate(geo, 0.1, show=show)

    geo.plot()

    ref_norms = [4.1468030858462305e02, 2.1861920662017866e01, 2.0]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-5)
    return X, cells


if __name__ == "__main__":
    X, cells = test(show=False)
    save("polygon.png", X, cells)
