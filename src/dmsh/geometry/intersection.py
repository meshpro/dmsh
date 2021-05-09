import numpy as np

from ..helpers import find_feature_points
from .geometry import Geometry


class Intersection(Geometry):
    def __init__(self, geometries):
        super().__init__()
        self.geometries = geometries
        self.bounding_box = [
            np.max([geo.bounding_box[0] for geo in geometries]),
            np.min([geo.bounding_box[1] for geo in geometries]),
            np.max([geo.bounding_box[2] for geo in geometries]),
            np.min([geo.bounding_box[3] for geo in geometries]),
        ]

        self.feature_points = find_feature_points(geometries)
        # filter out the feature points outside the intersection
        self.feature_points = self.feature_points[
            np.all(
                [geo.dist(self.feature_points.T) < 1.0e-10 for geo in geometries],
                axis=0,
            )
        ]

        self.paths = [path for geo in self.geometries for path in geo.paths]

    def dist(self, x):
        return np.max([geo.dist(x) for geo in self.geometries], axis=0)

    def boundary_step(self, x, tol=1.0e-12, max_steps=100):
        # step for the is_inside with the smallest value
        x = np.asarray(x)
        alpha = np.array([geo.dist(x) for geo in self.geometries])
        step = 0
        while np.any(np.abs(np.max(alpha, axis=0)) > tol):
            assert step <= max_steps, "Exceeded maximum number of boundary steps."
            step += 1

            # If the point has a positive geo distance, it is outside of the domain. In
            # this case, move it to the geo boundary with the largest distance.
            # If the point is strictly inside all geometries, move it to the closest
            # geometry boundary.
            # Both of these cases correspond to finding the domain with the max dist
            # value.
            mask = np.any(alpha > tol, axis=0) | np.all(alpha < -tol, axis=0)
            x_tmp = x[:, mask]
            alpha_pos = alpha[:, mask]
            idx = np.argmax(alpha_pos, axis=0)
            for k, geo in enumerate(self.geometries):
                if np.any(idx == k):
                    x_tmp[:, idx == k] = geo.boundary_step(x_tmp[:, idx == k])
            x[:, mask] = x_tmp

            alpha = np.array([geo.dist(x) for geo in self.geometries])
        return x
