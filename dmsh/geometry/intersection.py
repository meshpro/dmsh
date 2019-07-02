import numpy

from ..helpers import find_feature_points


class Intersection:
    def __init__(self, geometries):
        self.geometries = geometries
        self.bounding_box = [
            numpy.max([geo.bounding_box[0] for geo in geometries]),
            numpy.min([geo.bounding_box[1] for geo in geometries]),
            numpy.max([geo.bounding_box[2] for geo in geometries]),
            numpy.min([geo.bounding_box[3] for geo in geometries]),
        ]

        self.feature_points = find_feature_points(geometries)
        return

    def plot(self, color="b"):
        for geo in self.geometries:
            geo.plot()
        return

    def dist(self, x):
        return numpy.max([geo.dist(x) for geo in self.geometries], axis=0)

    def boundary_step(self, x, tol=1.0e-12):
        # step for the is_inside with the smallest value
        alpha = numpy.array([geo.dist(x) for geo in self.geometries])
        while numpy.any(alpha > tol):
            # Only consider the nodes which are truly outside of the domain
            has_pos = numpy.any(alpha > tol, axis=0)
            x_pos = x[:, has_pos]
            alpha_pos = alpha[:, has_pos]

            alpha_pos[alpha_pos < tol] = numpy.inf
            idx = numpy.argmin(alpha_pos, axis=0)
            for k, geo in enumerate(self.geometries):
                if numpy.any(idx == k):
                    x_pos[:, idx == k] = geo.boundary_step(x_pos[:, idx == k])

            x[:, has_pos] = x_pos
            alpha = numpy.array([geo.dist(x) for geo in self.geometries])
        return x
