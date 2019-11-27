import numpy

from .geometry import Geometry
from ..helpers import find_feature_points


class Union(Geometry):
    def __init__(self, geometries):
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
        return

    def dist(self, x):
        return numpy.min([geo.dist(x) for geo in self.geometries], axis=0)

    def boundary_step(self, x):
        # Step to the geometry with the smallest distance. The new point is not always
        # on the domain boundary.
        out = x.copy()
        alpha = numpy.array([geo.dist(x) for geo in self.geometries])
        idx = numpy.argmin(numpy.abs(alpha), axis=0)
        for k, geo in enumerate(self.geometries):
            j = (idx == k)
            if numpy.any(j):
                out[:, j] = geo.boundary_step(x[:, j])
        return out
