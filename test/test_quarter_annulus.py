import meshplex
import numpy
# from helpers import assert_norm_equality

import dmsh


def test_quarter_annulus(h):
    disk0 = dmsh.Circle([0.0, 0.0], 0.25)
    disk1 = dmsh.Circle([0.0, 0.0], 1.0)
    diff0 = dmsh.Difference(disk1, disk0)

    rect = dmsh.Rectangle(0.0, 1.0, 0.0, 1.0)
    quarter = dmsh.Intersection([diff0, rect])

    points, cells = dmsh.generate_mesh(
        domain=quarter,
        edge_size=lambda x: h + 0.10 * numpy.abs(disk0.eval(x)),
        h0=h,
        tol=1.0e-10
    )
    return points, cells


if __name__ == "__main__":
    test_quarter_annulus(0.02)
