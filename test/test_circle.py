import meshplex
import numpy
import pytest
from helpers import assert_norm_equality, save

import dmsh


@pytest.mark.parametrize(
    "radius,ref_norms",
    [(0.1, [327.95194, 14.263721, 1.0]), (0.4, [18.899253166, 3.70111746, 1.0])],
)
def test_circle(radius, ref_norms, show=False):
    geo = dmsh.Circle([0.0, 0.0], 1.0)
    X, cells = dmsh.generate(geo, radius, show=show)

    # make sure the origin is part of the mesh
    assert numpy.sum(numpy.einsum("ij,ij->i", X, X) < 1.0e-10) == 1

    assert_norm_equality(X.flatten(), ref_norms, 1.0e-5)
    return X, cells


def test_degenerate_circle():
    # with this target edge length, dmsh once produced weird results near the boundary
    target_edge_length = 0.07273
    geo = dmsh.Circle([0.0, 0.0], 1.0)
    X, cells = dmsh.generate(geo, target_edge_length, show=True)

    mesh = meshplex.MeshTri(X, cells)
    min_q = numpy.min(mesh.q_radius_ratio)
    assert min_q > 0.5, f"min cell quality: {min_q:.3f}"


if __name__ == "__main__":
    X, cells = test_circle(0.1, [327.95194, 14.263721, 1.0], show=False)
    save("circle.png", X, cells)
