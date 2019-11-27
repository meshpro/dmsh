import dmsh
from helpers import assert_norm_equality


def test(show=False):
    geo = dmsh.Difference(
        dmsh.Rectangle(0.0, 5.0, 0.0, 5.0),
        dmsh.Polygon([[1, 1], [4, 1], [4, 4], [1, 4]]),
    )
    X, cells = dmsh.generate(geo, 1.0, show=show, tol=1.0e-3)

    assert_norm_equality(
        X.flatten(), [1.3248809999934363e+02, 2.2652404941660635e+01, 5.0], 1.0e-12
    )


if __name__ == "__main__":
    test(show=True)
