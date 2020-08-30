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

    def boundary_step(self, x, tol=1.0e-5, max_steps=10):
        # This stepper uses a heuristic that doesn't always land on the closet point in
        # the boundary. It's good enough for now though.
        x = numpy.asarray(x)
        assert x.shape[0] == 2

        is_one_dimensional = False
        if len(x.shape) == 1:
            is_one_dimensional = True
            x = x.reshape(-1, 1)

        out = x.copy()

        # At the end of this, min(dist(domains)) must be 0 for all x, so each x either
        # sits outside or on the boundary of each domain. Idea: Use the domain with the
        # min value for stepping, and do that until all x are on the boundary.
        dists = numpy.array([geo.dist(out) for geo in self.geometries])
        dist = numpy.min(dists, axis=0)
        needs_stepping = (dist < -tol) | (tol < dist)

        k = 0
        while numpy.any(needs_stepping):
            assert k <= max_steps, "Too many union boundary steps"
            # Step to the boundary of the domain with the smallest value. (Either it's
            # far inside or domain or, if outside of the union, the closest to a
            # boundary.)
            idx = numpy.argmin(dists[:, needs_stepping], axis=0)
            for i, geo in enumerate(self.geometries):
                j = idx == i
                if numpy.any(j):
                    out[:, j] = geo.boundary_step(out[:, j])

            dists = numpy.array([geo.dist(out) for geo in self.geometries])
            dist = numpy.min(dists, axis=0)
            needs_stepping = (dist < -tol) | (tol < dist)
            k += 1

        if is_one_dimensional:
            out = out.reshape(-1)
        return out
