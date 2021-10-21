from typing import Tuple

import numpy as np

from .geometry import Geometry


class CirclePath:
    def __init__(self, x0: Tuple[float, float], r: float):
        self.x0 = x0
        self.r = r

    def p(self, t):
        v = np.array([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])
        return ((self.r * v).T + self.x0).T

    def dp_dt(self, t):
        return (
            self.r
            * 2
            * np.pi
            * np.array([-np.sin(2 * np.pi * t), np.cos(2 * np.pi * t)])
        )


class Circle(Geometry):
    def __init__(self, x0, r):
        self.x0 = x0
        self.r = r
        bounding_box = [x0[0] - r, x0[0] + r, x0[1] - r, x0[1] + r]
        self.paths = [CirclePath(x0, r)]
        feature_points = np.array([[], []]).T
        super().__init__(bounding_box, feature_points)

    def dist(self, x):
        assert x.shape[0] == 2
        y = (x.T - self.x0).T
        return np.sqrt(np.einsum("i...,i...->...", y, y)) - self.r

    def boundary_step(self, x):
        # simply project onto the circle
        y = (x.T - self.x0).T
        r = np.sqrt(np.einsum("ij,ij->j", y, y))
        return ((y / r * self.r).T + self.x0).T

    def plot(self, level_set=True):
        import matplotlib.pyplot as plt

        if level_set:
            X, Y, Z = self._get_xyz()
            alpha = np.max(np.abs(Z))
            cf = plt.contourf(
                X, Y, Z, levels=20, cmap=plt.cm.coolwarm, vmin=-alpha, vmax=alpha
            )
            plt.colorbar(cf)

        circle1 = plt.Circle(self.x0, self.r, color="k", fill=False)
        plt.gca().add_patch(circle1)

        plt.gca().set_aspect("equal")
