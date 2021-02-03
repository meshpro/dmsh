import meshplex
import numpy as np
import pytest
from helpers import assert_norm_equality

import dmsh


@pytest.mark.parametrize(
    "radius,ref_norms",
    [
        (0.1, [3.2592107070061820e02, 1.4190745248684369e01, 1.0000000000000000e00]),
        (0.4, [18.899253166, 3.70111746, 1.0]),
    ],
)
def test_circle(radius, ref_norms, show=False):
    geo = dmsh.Circle([0.0, 0.0], 1.0)
    X, cells = dmsh.generate(geo, radius, show=show, max_steps=100)
    meshplex.MeshTri(X, cells).show()

    # make sure the origin is part of the mesh
    assert np.sum(np.einsum("ij,ij->i", X, X) < 1.0e-6) == 1

    assert_norm_equality(X.flatten(), ref_norms, 1.0e-5)
    return X, cells


# with these target edge lengths, dmsh once produced weird results near the boundary
@pytest.mark.parametrize(
    "target_edge_length", [0.07273, 0.07272, 0.07271, 0.0711, 0.03591]
)
def test_degenerate_circle(target_edge_length):
    geo = dmsh.Circle([0.0, 0.0], 1.0)
    X, cells = dmsh.generate(geo, target_edge_length, show=False, max_steps=200)

    mesh = meshplex.MeshTri(X, cells)
    min_q = np.min(mesh.q_radius_ratio)
    assert min_q > 0.5, f"min cell quality: {min_q:.3f}"


def test_boundary_step():
    geo = dmsh.Circle([0.1, 0.2], 1.0)
    np.random.seed(0)
    pts = np.random.uniform(-1.0, 1.0, (2, 100))
    pts = geo.boundary_step(pts)
    tol = 1.0e-12
    assert np.all(np.abs(geo.dist(pts)) < tol)


if __name__ == "__main__":
    test_boundary_step()
