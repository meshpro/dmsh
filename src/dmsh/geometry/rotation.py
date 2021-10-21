import numpy as np

from .geometry import Geometry


class Rotation(Geometry):
    def __init__(self, geometry, angle):
        self.geometry = geometry

        self.R = np.array(
            [
                [+np.cos(angle), -np.sin(angle)],
                [+np.sin(angle), +np.cos(angle)],
            ]
        )
        self.R_inv = np.array(
            [
                [+np.cos(angle), +np.sin(angle)],
                [-np.sin(angle), +np.cos(angle)],
            ]
        )

        # bounding box
        bb = geometry.bounding_box
        corners = np.array(
            [[bb[0], bb[2]], [bb[1], bb[2]], [bb[1], bb[3]], [bb[0], bb[3]]]
        )
        rotated_corners = np.dot(self.R, corners.T)
        bounding_box = [
            np.min(rotated_corners[0]),
            np.max(rotated_corners[0]),
            np.min(rotated_corners[1]),
            np.max(rotated_corners[1]),
        ]
        super().__init__(bounding_box, feature_points=[])

    def dist(self, x):
        return self.geometry.dist(np.dot(self.R_inv, x))

    def boundary_step(self, x):
        y = np.dot(self.R_inv, x)
        y2 = self.geometry.boundary_step(y)
        return np.dot(self.R, y2)
