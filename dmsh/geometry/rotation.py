import numpy

from .geometry import Geometry


class Rotation(Geometry):
    def __init__(self, geometry, angle):
        super().__init__()
        self.geometry = geometry

        self.R = numpy.array(
            [
                [+numpy.cos(angle), -numpy.sin(angle)],
                [+numpy.sin(angle), +numpy.cos(angle)],
            ]
        )
        self.R_inv = numpy.array(
            [
                [+numpy.cos(angle), +numpy.sin(angle)],
                [-numpy.sin(angle), +numpy.cos(angle)],
            ]
        )

        # bounding box
        bb = geometry.bounding_box
        corners = numpy.array(
            [[bb[0], bb[2]], [bb[1], bb[2]], [bb[1], bb[3]], [bb[0], bb[3]]]
        )
        rotated_corners = numpy.dot(self.R, corners.T)
        self.bounding_box = [
            numpy.min(rotated_corners[0]),
            numpy.max(rotated_corners[0]),
            numpy.min(rotated_corners[1]),
            numpy.max(rotated_corners[1]),
        ]
        self.feature_points = numpy.array([])
        return

    def dist(self, x):
        return self.geometry.dist(numpy.dot(self.R_inv, x))

    def boundary_step(self, x):
        y = numpy.dot(self.R_inv, x)
        y2 = self.geometry.boundary_step(y)
        return numpy.dot(self.R, y2)
