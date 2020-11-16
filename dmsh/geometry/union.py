import numpy

from ..helpers import find_feature_points
from .geometry import Geometry


class Union(Geometry):
    def __init__(self, geometries):
        super().__init__()
        self.geometries = geometries
        self.bounding_box = [
            numpy.min([geo.bounding_box[0] for geo in geometries]),
            numpy.max([geo.bounding_box[1] for geo in geometries]),
            numpy.min([geo.bounding_box[2] for geo in geometries]),
            numpy.max([geo.bounding_box[3] for geo in geometries]),
        ]

        fp = [geo.feature_points for geo in geometries]
        fp.append(find_feature_points(geometries))
        self.feature_points = numpy.concatenate(fp)

        # Only keep the feature points on the outer boundary
        alpha = numpy.array([geo.dist(self.feature_points.T) for geo in geometries])
        tol = 1.0e-5
        is_on_boundary = numpy.all(alpha > -tol, axis=0)
        self.feature_points = self.feature_points[is_on_boundary]

        self.paths = [path for geo in self.geometries for path in geo.paths]

    def dist(self, x):
        return numpy.min([geo.dist(x) for geo in self.geometries], axis=0)

    def boundary_step(self, x, tol=1.0e-12, max_steps=100):
        # step for the is_inside with the smallest value
        x = numpy.asarray(x)
        alpha = numpy.array([geo.dist(x) for geo in self.geometries])
        step = 0
        while numpy.any(numpy.abs(numpy.min(alpha, axis=0)) > tol):
            assert step <= max_steps, "Exceeded maximum number of boundary steps."
            step += 1

            # If the point has a positive geo distance, it is outside of the domain. In
            # this case, move it to the geo boundary with the smallest distance.
            # If the point is strictly inside all geometries, move it to the furthest
            # geometry boundary.
            mask = numpy.all(alpha > tol, axis=0) | numpy.any(alpha < -tol, axis=0)
            x_tmp = x[:, mask]
            idx = numpy.argmin(alpha[:, mask], axis=0)
            for k, geo in enumerate(self.geometries):
                if numpy.any(idx == k):
                    x_tmp[:, idx == k] = geo.boundary_step(x_tmp[:, idx == k])
            x[:, mask] = x_tmp

            alpha = numpy.array([geo.dist(x) for geo in self.geometries])
        return x
