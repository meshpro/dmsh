import numpy
import pytest
from helpers import assert_norm_equality, save

import dmsh


@pytest.mark.parametrize(
    "radius,ref_norms",
    [(0.1, [327.95194, 14.263721, 1.0]), (0.4, [18.899253166, 3.70111746, 1.0])],
)
def test_custom(radius, ref_norms, show=False):

    class MyShape(dmsh.Geometry):
        def __init__(self):
            super().__init__()
            self.r = 1.0
            self.x0 = [0.0, 0.0]
            self.bounding_box = [-1.0, 1.0, -1.0, 1.0]
            self.feature_points = numpy.array([[], []]).T

        def dist(self, x):
            assert x.shape[0] == 2
            y = (x.T - self.x0).T
            return numpy.sqrt(numpy.einsum("i...,i...->...", y, y)) - self.r

        def boundary_step(self, x):
            # simply project onto the circle
            y = (x.T - self.x0).T
            r = numpy.sqrt(numpy.einsum("ij,ij->j", y, y))
            return ((y / r * self.r).T + self.x0).T

    geo = MyShape()
    X, cells = dmsh.generate(geo, radius, show=show)

    # make sure the origin is part of the mesh
    assert numpy.sum(numpy.einsum("ij,ij->i", X, X) < 1.0e-10) == 1

    assert_norm_equality(X.flatten(), ref_norms, 1.0e-5)
    return X, cells


if __name__ == "__main__":
    X, cells = test_custom(0.1, [327.95194, 14.263721, 1.0], show=False)
    save("circle.png", X, cells)
