from helpers import assert_norm_equality, save

import dmsh


def test(show=False):
    """
    Generate the x y x y coordinates

    Args:
        show: (bool): write your description
    """
    geo = dmsh.Rectangle(0.0, 1.0, 0.0, 1.0)

    # p0 = dmsh.Path([[0.0, 0.0]])
    p1 = dmsh.Path([[0.4, 0.6], [0.6, 0.4]])

    def edge_size(x):
        """
        Returns the size of x.

        Args:
            x: (todo): write your description
        """
        return 0.03 + 0.1 * p1.dist(x)

    X, cells = dmsh.generate(geo, edge_size, show=show, tol=1.0e-10)

    ref_norms = [3.8844963377954718e02, 1.5689926870601147e01, 1.0000000000000000e00]
    assert_norm_equality(X.flatten(), ref_norms, 1.0e-3)
    return X, cells


if __name__ == "__main__":
    X, cells = test(show=False)
    save("refinement_line.png", X, cells)
