import numpy


class Scaling:
    def __init__(self, geometry, alpha):
        self.geometry = geometry
        self.alpha = alpha
        self.bounding_box = alpha * numpy.array(geometry.bounding_box)
        self.feature_points = numpy.array([])
        return

    def dist(self, x):
        return self.geometry.dist(x / self.alpha)

    def boundary_step(self, x):
        return self.geometry.boundary_step(x / self.alpha) * self.alpha
