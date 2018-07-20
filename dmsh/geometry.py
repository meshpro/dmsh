# -*- coding: utf-8 -*-
#
import numpy
import polypy

from .helpers import multi_newton, find_feature_points


class Stretch(object):
    def __init__(self, geometry, v):
        self.geometry = geometry
        self.alpha = numpy.sqrt(numpy.dot(v, v))
        self.v = v / self.alpha

        # bounding box
        bb = geometry.bounding_box
        corners = numpy.array(
            [[bb[0], bb[2]], [bb[1], bb[2]], [bb[1], bb[3]], [bb[0], bb[3]]]
        )
        vx = numpy.multiply.outer(numpy.dot(self.v, corners.T), self.v)
        stretched_corners = (vx * self.alpha + (corners - vx)).T
        self.bounding_box = [
            numpy.min(stretched_corners[0]),
            numpy.max(stretched_corners[0]),
            numpy.min(stretched_corners[1]),
            numpy.max(stretched_corners[1]),
        ]
        self.feature_points = numpy.array([])
        return

    def plot(self):
        return

    def isinside(self, x):
        # scale the component of x in direction v by 1/alpha
        vx = numpy.multiply.outer(numpy.dot(self.v, x), self.v)
        y = vx / self.alpha + (x.T - vx)
        return self.geometry.isinside(y.T)

    def boundary_step(self, x):
        vx = numpy.multiply.outer(numpy.dot(self.v, x), self.v)
        y = vx / self.alpha + (x.T - vx)
        y2 = self.geometry.boundary_step(y.T)
        vy2 = numpy.multiply.outer(numpy.dot(self.v, y2), self.v)
        return (vy2 * self.alpha + (y2.T - vy2)).T


class Scaling(object):
    def __init__(self, geometry, alpha):
        self.geometry = geometry
        self.alpha = alpha
        self.bounding_box = alpha * numpy.array(geometry.bounding_box)
        self.feature_points = numpy.array([])
        return

    def plot(self):
        return

    def isinside(self, x):
        return self.geometry.isinside(x / self.alpha)

    def boundary_step(self, x):
        return self.geometry.boundary_step(x / self.alpha) * self.alpha


class Translation(object):
    def __init__(self, geometry, v):
        self.geometry = geometry
        self.v = v

        self.bounding_box = [
            geometry.bounding_box[0] + v[0],
            geometry.bounding_box[1] + v[0],
            geometry.bounding_box[2] + v[1],
            geometry.bounding_box[3] + v[1],
        ]
        self.feature_points = numpy.array([])
        return

    def plot(self):
        return

    def isinside(self, x):
        return self.geometry.isinside((x.T - self.v).T)

    def boundary_step(self, x):
        return (self.geometry.boundary_step((x.T - self.v).T).T + self.v).T


class Rotation(object):
    def __init__(self, geometry, angle):
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

    def plot(self):
        return

    def isinside(self, x):
        return self.geometry.isinside(numpy.dot(self.R_inv, x))

    def boundary_step(self, x):
        y = numpy.dot(self.R_inv, x)
        y2 = self.geometry.boundary_step(y)
        return numpy.dot(self.R, y2)


class Union(object):
    def __init__(self, geometries):
        self.geometries = geometries
        self.bounding_box = [
            numpy.min([geo.bounding_box[0] for geo in geometries]),
            numpy.max([geo.bounding_box[1] for geo in geometries]),
            numpy.min([geo.bounding_box[2] for geo in geometries]),
            numpy.max([geo.bounding_box[3] for geo in geometries]),
        ]

        self.feature_points = find_feature_points(geometries)
        return

    def plot(self):
        for geo in self.geometries:
            geo.plot()
        return

    def isinside(self, x):
        return numpy.min([geo.isinside(x) for geo in self.geometries], axis=0)

    def boundary_step(self, x):
        # step for the is_inside with the smallest value
        alpha = numpy.array([geo.isinside(x) for geo in self.geometries])
        alpha[alpha < 0] = numpy.inf
        idx = numpy.argmin(alpha, axis=0)
        for k, geo in enumerate(self.geometries):
            if numpy.any(idx == k):
                x[:, idx == k] = geo.boundary_step(x[:, idx == k])
        return x


class Intersection(object):
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

    def isinside(self, x):
        return numpy.max([geo.isinside(x) for geo in self.geometries], axis=0)

    def boundary_step(self, x, tol=1.0e-12):
        # step for the is_inside with the smallest value
        alpha = numpy.array([geo.isinside(x) for geo in self.geometries])
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
            alpha = numpy.array([geo.isinside(x) for geo in self.geometries])
        return x


class Difference(object):
    def __init__(self, geo0, geo1):
        self.geo0 = geo0
        self.geo1 = geo1
        self.bounding_box = geo0.bounding_box
        self.feature_points = find_feature_points([geo0, geo1])
        return

    def plot(self, color="b"):
        self.geo0.plot()
        self.geo1.plot()
        return

    def isinside(self, x):
        return numpy.max([self.geo0.isinside(x), -self.geo1.isinside(x)], axis=0)

    def boundary_step(self, x, tol=1.0e-12):
        # step for the is_inside with the smallest value
        alpha0 = self.geo0.isinside(x)
        alpha1 = self.geo1.isinside(x)
        while numpy.any(alpha0 > tol) or numpy.any(alpha1 < -tol):
            idx0 = alpha0 > tol
            if numpy.any(idx0):
                x[:, idx0] = self.geo0.boundary_step(x[:, idx0])
                alpha0 = self.geo0.isinside(x)
                continue

            idx1 = alpha1 < -tol
            if numpy.any(idx1):
                x[:, idx1] = self.geo1.boundary_step(x[:, idx1])
                alpha1 = self.geo1.isinside(x)
                continue
        return x


class Ellipse(object):
    def __init__(self, x0, a, b):
        self.x0 = x0
        self.a = a
        self.b = b
        self.bounding_box = [x0[0] - a, x0[0] + a, x0[1] - b, x0[1] + b]
        self.feature_points = numpy.array([])
        return

    def plot(self, color="b"):
        import matplotlib.pyplot as plt

        t = numpy.linspace(0.0, 2 * numpy.pi, 100)
        plt.plot(
            self.x0[0] + self.a * numpy.cos(t),
            self.x0[1] + self.b * numpy.sin(t),
            "-",
            color=color,
        )
        return

    def isinside(self, x):
        assert x.shape[0] == 2
        return (
            ((x[0] - self.x0[0]) / self.a) ** 2
            + ((x[1] - self.x0[1]) / self.b) ** 2
            - 1.0
        )

    def _boundary_step(self, x):
        ax = (x[0] - self.x0[0]) / self.a
        ay = (x[1] - self.x0[1]) / self.b

        alpha = ax ** 2 + ay ** 2 - 1.0
        jac = numpy.array([4 * alpha * ax / self.a, 4 * alpha * ay / self.b])

        dalpha_dx = 2 * ax / self.a
        dalpha_dy = 2 * ay / self.b
        hess = numpy.array(
            [
                [
                    4 * dalpha_dx * ax / self.a + 4 * alpha / self.a ** 2,
                    4 * dalpha_dy * ax / self.a,
                ],
                [
                    4 * dalpha_dx * ay / self.b,
                    4 * dalpha_dy * ay / self.b + 4 * alpha / self.b ** 2,
                ],
            ]
        )

        p = -numpy.linalg.solve(numpy.moveaxis(hess, -1, 0), jac.T)
        return x + p.T

    def boundary_step(self, x):
        return multi_newton(
            x.T, self.isinside, self._boundary_step, 1.0e-10, max_num_steps=10
        ).T


class CirclePath(object):
    def __init__(self, x0, r):
        self.x0 = x0
        self.r = r
        return

    def p(self, t):
        v = numpy.array([numpy.cos(2 * numpy.pi * t), numpy.sin(2 * numpy.pi * t)])
        return ((self.r * v).T + self.x0).T

    def dp_dt(self, t):
        return (
            self.r
            * 2
            * numpy.pi
            * numpy.array([-numpy.sin(2 * numpy.pi * t), numpy.cos(2 * numpy.pi * t)])
        )


class Circle(object):
    def __init__(self, x0, r):
        self.x0 = x0
        self.r = r
        self.bounding_box = [x0[0] - r, x0[0] + r, x0[1] - r, x0[1] + r]
        self.paths = [CirclePath(x0, r)]
        self.feature_points = numpy.array([])
        return

    def plot(self, color="b"):
        import matplotlib.pyplot as plt

        t = numpy.linspace(0.0, 2 * numpy.pi, 100)
        plt.plot(
            self.x0[0] + self.r * numpy.cos(t),
            self.x0[1] + self.r * numpy.sin(t),
            "-",
            color=color,
        )
        return

    def isinside(self, x):
        assert x.shape[0] == 2
        return (
            ((x[0] - self.x0[0]) / self.r) ** 2
            + ((x[1] - self.x0[1]) / self.r) ** 2
            - 1.0
        )

    def boundary_step(self, x):
        # simply project onto the circle
        x = (x.T - self.x0).T
        r = numpy.sqrt(numpy.einsum("ij,ij->j", x, x))
        return ((x / r * self.r).T + self.x0).T


class Polygon(object):
    def __init__(self, points):
        points = numpy.array(points)
        self.bounding_box = [
            numpy.min(points[:, 0]),
            numpy.max(points[:, 0]),
            numpy.min(points[:, 1]),
            numpy.max(points[:, 1]),
        ]
        self.polygon = polypy.Polygon(points)
        self.feature_points = points
        return

    def plot(self):
        self.polygon.plot()
        return

    def isinside(self, x):
        return self.polygon.signed_squared_distance(x.T)

    def boundary_step(self, x):
        return self.polygon.closest_points(x.T).T


class Rectangle(Polygon):
    def __init__(self, x0, x1, y0, y1):
        points = numpy.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        super(Rectangle, self).__init__(points)
        return


class HalfSpacePath(object):
    def __init__(self, normal, alpha):
        self.normal = normal
        self.alpha = alpha

        self.tangent = numpy.array([-self.normal[1], self.normal[0]])

        # One point on the line:
        self.v = self.normal / numpy.dot(self.normal, self.normal) * self.alpha
        return

    def p(self, t):
        """This parametrization of the line is (inf, inf) for t=0 and t=1.
        """
        # Don't warn on division by 0
        with numpy.errstate(divide="ignore"):
            out = (
                numpy.multiply.outer(self.tangent, (2 * t - 1) / t / (1 - t)).T + self.v
            ).T
        return out

    def dp_dt(self, t):
        with numpy.errstate(divide="ignore"):
            dt = 1 / t ** 2 + 1 / (1 - t) ** 2
        return numpy.multiply.outer(self.tangent, dt)


class HalfSpace(object):
    def __init__(self, normal, alpha):
        self.normal = normal
        self.alpha = alpha

        self.bounding_box = [-numpy.inf, +numpy.inf, -numpy.inf, +numpy.inf]
        self.feature_points = numpy.array([])

        self.paths = [HalfSpacePath(normal, alpha)]
        return

    def plot(self):
        return

    def isinside(self, x):
        assert x.shape[0] == 2
        return self.alpha - numpy.dot(self.normal, x)

    def boundary_step(self, x):
        beta = self.alpha - numpy.dot(self.normal, x) / numpy.dot(
            self.normal, self.normal
        )
        return x + numpy.multiply.outer(self.normal, beta)
