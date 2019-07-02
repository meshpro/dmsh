import numpy

from ..helpers import find_feature_points


class Union:
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
        return

    def plot(self):
        for geo in self.geometries:
            geo.plot()
        return

    def dist(self, x):
        return numpy.min([geo.dist(x) for geo in self.geometries], axis=0)

    def boundary_step(self, x):
        # step for the is_inside with the smallest value
        alpha = numpy.array([geo.dist(x) for geo in self.geometries])
        alpha[alpha < 0] = numpy.inf
        idx = numpy.argmin(alpha, axis=0)
        for k, geo in enumerate(self.geometries):
            if numpy.any(idx == k):
                x[:, idx == k] = geo.boundary_step(x[:, idx == k])
        return x
