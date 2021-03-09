import meshio
import numpy as np
import optimesh

import dmsh


def save(X, cells, filename):
    meshio.Mesh(X, {"triangle": cells}).write(
        filename, stroke="#969696", force_width=100, stroke_width=0.5
    )


geo = dmsh.Circle([0.0, 0.0], 1.0)
X, cells = dmsh.generate(geo, 0.1)
# optionally optimize the mesh
X, cells = optimesh.optimize_points_cells(X, cells, "CVT (full)", 1.0e-10, 100)
save(X, cells, "circle.svg")


geo = dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0)
X, cells = dmsh.generate(geo, 0.1)
save(X, cells, "rectangle.svg")

geo = dmsh.Polygon(
    [
        [0.0, 0.0],
        [1.1, 0.0],
        [1.2, 0.5],
        [0.7, 0.6],
        [2.0, 1.0],
        [1.0, 2.0],
        [0.5, 1.5],
    ]
)
X, cells = dmsh.generate(geo, 0.1)
save(X, cells, "polygon.svg")

geo = dmsh.Difference(dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0))
X, cells = dmsh.generate(geo, 0.1)
save(X, cells, "moon.svg")


geo = dmsh.Difference(
    dmsh.Circle([0.0, 0.0], 1.0),
    dmsh.Polygon([[0.0, 0.0], [1.5, 0.4], [1.5, -0.4]]),
)
X, cells = dmsh.generate(geo, 0.1, tol=1.0e-10)
save(X, cells, "pacman.svg")


r = dmsh.Rectangle(-1.0, +1.0, -1.0, +1.0)
c = dmsh.Circle([0.0, 0.0], 0.3)
geo = dmsh.Difference(r, c)
X, cells = dmsh.generate(geo, lambda pts: np.abs(c.dist(pts)) / 5 + 0.05, tol=1.0e-10)
save(X, cells, "rectangle-hole-refinement.svg")


geo = dmsh.Union([dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0)])
X, cells = dmsh.generate(geo, 0.15)
save(X, cells, "union-circles.svg")


geo = dmsh.Union(
    [dmsh.Rectangle(-1.0, +0.5, -1.0, +0.5), dmsh.Rectangle(-0.5, +1.0, -0.5, +1.0)]
)
X, cells = dmsh.generate(geo, 0.15)
save(X, cells, "union-rectangles.svg")


angles = np.pi * np.array([3.0 / 6.0, 7.0 / 6.0, 11.0 / 6.0])
geo = dmsh.Union(
    [
        dmsh.Circle([np.cos(angles[0]), np.sin(angles[0])], 1.0),
        dmsh.Circle([np.cos(angles[1]), np.sin(angles[1])], 1.0),
        dmsh.Circle([np.cos(angles[2]), np.sin(angles[2])], 1.0),
    ]
)
X, cells = dmsh.generate(geo, 0.15)
save(X, cells, "union-three-circles.svg")


geo = dmsh.Intersection([dmsh.Circle([0.0, -0.5], 1.0), dmsh.Circle([0.0, +0.5], 1.0)])
X, cells = dmsh.generate(geo, 0.1, tol=1.0e-10)
save(X, cells, "intersection-circles.svg")


angles = np.pi * np.array([3.0 / 6.0, 7.0 / 6.0, 11.0 / 6.0])
geo = dmsh.Intersection(
    [
        dmsh.Circle([np.cos(angles[0]), np.sin(angles[0])], 1.5),
        dmsh.Circle([np.cos(angles[1]), np.sin(angles[1])], 1.5),
        dmsh.Circle([np.cos(angles[2]), np.sin(angles[2])], 1.5),
    ]
)
X, cells = dmsh.generate(geo, 0.1, tol=1.0e-10)
save(X, cells, "intersection-three-circles.svg")


geo = dmsh.Intersection(
    [
        dmsh.HalfSpace(np.sqrt(0.5) * np.array([1.0, 1.0]), 0.0),
        dmsh.Circle([0.0, 0.0], 1.0),
    ]
)
X, cells = dmsh.generate(geo, 0.1)
save(X, cells, "intersection-circle-halfspace.svg")


geo = dmsh.Rotation(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), 0.1 * np.pi)
X, cells = dmsh.generate(geo, 0.1, tol=1.0e-10)
save(X, cells, "rotation.svg")


geo = dmsh.Scaling(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), 2.0)
X, cells = dmsh.generate(geo, 0.1, tol=1.0e-5)
save(X, cells, "scaling.svg")


geo = dmsh.Rectangle(0.0, 1.0, 0.0, 1.0)
p1 = dmsh.Path([[0.4, 0.6], [0.6, 0.4]])
X, cells = dmsh.generate(geo, edge_size=lambda x: 0.03 + 0.1 * p1.dist(x), tol=1.0e-10)
save(X, cells, "local-refinement.svg")
