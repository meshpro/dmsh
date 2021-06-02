import numpy as np

from ..helpers import find_feature_points


class Geometry:
    def __init__(self):
        return

    def _get_xyz(self, nx=101, ny=101):
        x0, x1, y0, y1 = self.bounding_box
        w = x1 - x0
        h = x1 - x0
        x = np.linspace(x0 - w * 0.1, x1 + w * 0.1, nx)
        y = np.linspace(y0 - h * 0.1, y1 + h * 0.1, ny)
        X, Y = np.meshgrid(x, y)
        Z = self.dist(np.array([X, Y]))
        return X, Y, Z

    def _plot_level_set(self):
        import matplotlib.pyplot as plt

        X, Y, Z = self._get_xyz()
        alpha = np.max(np.abs(Z))
        cf = plt.contourf(
            X, Y, Z, levels=20, cmap=plt.cm.coolwarm, vmin=-alpha, vmax=alpha
        )
        plt.colorbar(cf)

    def plot(self, level_set=True):
        import matplotlib.pyplot as plt

        X, Y, Z = self._get_xyz()

        if level_set:
            alpha = np.max(np.abs(Z))
            cf = plt.contourf(
                X, Y, Z, levels=20, cmap=plt.cm.coolwarm, vmin=-alpha, vmax=alpha
            )
            plt.colorbar(cf)

        # mark the 0-level (the domain boundary)
        plt.contour(X, Y, Z, levels=[0.0], colors="k")

        plt.gca().set_aspect("equal")

    def show(self, *args, **kwargs):
        import matplotlib.pyplot as plt

        self.plot(*args, **kwargs)
        plt.show()

    def __add__(self, obj):
        if isinstance(obj, Geometry):
            return Union([self, obj])
        return Translation(self, obj)

    def __radd__(self, obj):
        return self.__add__(obj)

    def __sub__(self, obj):
        if isinstance(obj, Geometry):
            return Difference(self, obj)
        return Translation(self, -obj)

    def __and__(self, obj):
        return Intersection([self, obj])

    def __or__(self, obj):
        return Union([self, obj])

    def __mul__(self, alpha: float):
        return Scaling(self, alpha)

    def __rmul__(self, alpha: float):
        return self.__mul__(alpha)

    def stretch(self, obj):
        return Stretch(self, obj)


class Union(Geometry):
    def __init__(self, geometries):
        super().__init__()
        self.geometries = geometries
        self.bounding_box = [
            np.min([geo.bounding_box[0] for geo in geometries]),
            np.max([geo.bounding_box[1] for geo in geometries]),
            np.min([geo.bounding_box[2] for geo in geometries]),
            np.max([geo.bounding_box[3] for geo in geometries]),
        ]

        fp = [geo.feature_points for geo in geometries]
        fp.append(find_feature_points(geometries))
        self.feature_points = np.concatenate(fp)

        # Only keep the feature points on the outer boundary
        alpha = np.array([geo.dist(self.feature_points.T) for geo in geometries])
        tol = 1.0e-5
        is_on_boundary = np.all(alpha > -tol, axis=0)
        self.feature_points = self.feature_points[is_on_boundary]

        self.paths = [path for geo in self.geometries for path in geo.paths]

    def dist(self, x):
        return np.min([geo.dist(x) for geo in self.geometries], axis=0)

    def boundary_step(self, x, tol=1.0e-12, max_steps=100):
        # step for the is_inside with the smallest value
        x = np.asarray(x)
        alpha = np.array([geo.dist(x) for geo in self.geometries])
        step = 0
        while np.any(np.abs(np.min(alpha, axis=0)) > tol):
            assert step <= max_steps, "Exceeded maximum number of boundary steps."
            step += 1

            # If the point has a positive geo distance, it is outside of the domain. In
            # this case, move it to the geo boundary with the smallest distance.
            # If the point is strictly inside all geometries, move it to the furthest
            # geometry boundary.
            mask = np.all(alpha > tol, axis=0) | np.any(alpha < -tol, axis=0)
            x_tmp = x[:, mask]
            idx = np.argmin(alpha[:, mask], axis=0)
            for k, geo in enumerate(self.geometries):
                if np.any(idx == k):
                    x_tmp[:, idx == k] = geo.boundary_step(x_tmp[:, idx == k])
            x[:, mask] = x_tmp

            alpha = np.array([geo.dist(x) for geo in self.geometries])
        return x


class Stretch(Geometry):
    def __init__(self, geometry, v):
        super().__init__()
        self.geometry = geometry
        self.alpha = np.sqrt(np.dot(v, v))
        self.v = v / self.alpha

        # bounding box
        bb = geometry.bounding_box
        corners = np.array(
            [[bb[0], bb[2]], [bb[1], bb[2]], [bb[1], bb[3]], [bb[0], bb[3]]]
        )
        vx = np.multiply.outer(np.dot(self.v, corners.T), self.v)
        stretched_corners = (vx * self.alpha + (corners - vx)).T
        self.bounding_box = [
            np.min(stretched_corners[0]),
            np.max(stretched_corners[0]),
            np.min(stretched_corners[1]),
            np.max(stretched_corners[1]),
        ]
        self.feature_points = np.array([])

    def dist(self, x):
        # scale the component of x in direction v by 1/alpha
        x_shape = x.shape
        assert x.shape[0] == 2
        x = x.reshape(2, -1)
        vx = np.multiply.outer(np.dot(self.v, x), self.v)
        y = vx / self.alpha + (x.T - vx)
        y = y.T.reshape(x_shape)
        return self.geometry.dist(y)

    def boundary_step(self, x):
        vx = np.multiply.outer(np.dot(self.v, x), self.v)
        y = vx / self.alpha + (x.T - vx)
        y2 = self.geometry.boundary_step(y.T)
        vy2 = np.multiply.outer(np.dot(self.v, y2), self.v)
        return (vy2 * self.alpha + (y2.T - vy2)).T


class Difference(Geometry):
    def __init__(self, geo0, geo1):
        super().__init__()
        self.geo0 = geo0
        self.geo1 = geo1
        self.bounding_box = geo0.bounding_box

        fp = [geo0.feature_points, geo1.feature_points]
        fp.append(find_feature_points([geo0, geo1]))
        self.feature_points = np.concatenate(fp)

        # Only keep the feature points on the outer boundary
        alpha = self.dist(self.feature_points.T)
        tol = 1.0e-5
        is_on_boundary = (-tol < alpha) & (alpha < tol)
        self.feature_points = self.feature_points[is_on_boundary]

        self.paths = [path for geo in [geo0, geo1] for path in geo.paths]

    def dist(self, x):
        return np.max([self.geo0.dist(x), -self.geo1.dist(x)], axis=0)

    # Choose tolerance above sqrt(machine_eps). This is necessary as the polygon
    # dist() is only accurate to that precision.
    def boundary_step(self, x, tol=1.0e-12, max_steps=100):
        # Scale the tolerance with the domain diameter. This is necessary at least for
        # polygons where the distance calculation is flawed with round-off proportional
        # to the edge lengths.
        try:
            tol *= self.geo0.diameter
        except AttributeError:
            pass

        alpha = np.array([self.geo0.dist(x), -self.geo1.dist(x)])
        mask = np.any(alpha > tol, axis=0) | np.all(alpha < -tol, axis=0)

        step = 0
        while np.any(mask):
            assert step <= max_steps, "Exceeded maximum number of boundary steps."
            step += 1

            x_tmp = x[:, mask]
            idx = np.argmax(alpha[:, mask], axis=0)
            if np.any(idx == 0):
                x_tmp[:, idx == 0] = self.geo0.boundary_step(x_tmp[:, idx == 0])
            if np.any(idx == 1):
                x_tmp[:, idx == 1] = self.geo1.boundary_step(x_tmp[:, idx == 1])
            x[:, mask] = x_tmp

            alpha = np.array([self.geo0.dist(x), -self.geo1.dist(x)])
            mask = np.any(alpha > tol, axis=0) | np.all(alpha < -tol, axis=0)
        return x


class Translation(Geometry):
    def __init__(self, geometry, v):
        super().__init__()
        self.geometry = geometry
        self.v = v

        self.bounding_box = [
            geometry.bounding_box[0] + v[0],
            geometry.bounding_box[1] + v[0],
            geometry.bounding_box[2] + v[1],
            geometry.bounding_box[3] + v[1],
        ]
        self.feature_points = np.array([])
        return

    def dist(self, x):
        return self.geometry.dist((x.T - self.v).T)

    def boundary_step(self, x):
        return (self.geometry.boundary_step((x.T - self.v).T).T + self.v).T


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


class Scaling(Geometry):
    def __init__(self, geometry: Geometry, alpha: float):
        super().__init__()
        self.geometry = geometry
        self.alpha = alpha
        self.bounding_box = alpha * np.array(geometry.bounding_box)
        self.feature_points = np.array([])

    def dist(self, x):
        return self.geometry.dist(x / self.alpha)

    def boundary_step(self, x):
        return self.geometry.boundary_step(x / self.alpha) * self.alpha
