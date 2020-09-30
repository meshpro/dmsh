<p align="center">
  <a href="https://github.com/nschloe/dmsh"><img alt="dmsh" src="https://nschloe.github.io/dmsh/logo-with-text.svg" width="50%"></a>
  <p align="center">The worst mesh generator you'll ever use.</p>
</p>

[![PyPi Version](https://img.shields.io/pypi/v/dmsh.svg?style=flat-square)](https://pypi.org/project/dmsh)
[![Packaging status](https://repology.org/badge/tiny-repos/python:dmsh.svg)](https://repology.org/project/python:dmsh/versions)
[![PyPI pyversions](https://img.shields.io/pypi/pyversions/dmsh.svg?style=flat-square)](https://pypi.org/pypi/dmsh/)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/dmsh.svg?style=flat-square&logo=github&label=Stars&logoColor=white)](https://github.com/nschloe/dmsh)
[![PyPi downloads](https://img.shields.io/pypi/dm/dmsh.svg?style=flat-square)](https://pypistats.org/packages/dmsh)

[![gh-actions](https://img.shields.io/github/workflow/status/nschloe/dmsh/ci?style=flat-square)](https://github.com/nschloe/dmsh/actions?query=workflow%3Aci)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/dmsh.svg?style=flat-square)](https://codecov.io/gh/nschloe/dmsh)
[![LGTM](https://img.shields.io/lgtm/grade/python/github/nschloe/dmsh.svg?style=flat-square)](https://lgtm.com/projects/g/nschloe/dmsh)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg?style=flat-square)](https://github.com/psf/black)

Inspired by [distmesh](http://persson.berkeley.edu/distmesh/), dmsh can be slow,
requires a lot of memory, and isn't terribly robust either.

On the plus side,

 * it's got a user-friendly interface,
 * is pure Python (and hence easily installable on any system), and
 * it produces pretty high-quality meshes.

Combined with [optimesh](https://github.com/nschloe/optimesh), dmsh produces the
highest-quality 2D meshes in the west.

### Examples

#### Primitives

![circle](https://nschloe.github.io/dmsh/circle.png) | ![rectangle](https://nschloe.github.io/dmsh/rectangle.png) | ![polygon](https://nschloe.github.io/dmsh/polygon.png)
|:---:|:---:|:---:|

```python
import dmsh

geo = dmsh.Circle([0.0, 0.0], 1.0)
X, cells = dmsh.generate(geo, 0.1)

# optionally optimize the mesh
import optimesh

X, cells = optimesh.cvt.quasi_newton_uniform_full(X, cells, 1.0e-10, 100)

# visualize the mesh
dmsh.helpers.show(X, cells, geo)

# and write it to a file
import meshio

meshio.write_points_cells("circle.vtk", X, {"triangle": cells})
```

```python
import dmsh

geo = dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0)
X, cells = dmsh.generate(geo, 0.1)
```

```python
import dmsh

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
```

#### Combinations

##### Difference

![difference](https://nschloe.github.io/dmsh/difference.png) | ![pacman](https://nschloe.github.io/dmsh/pacman.png) | ![square_hole_refined](https://nschloe.github.io/dmsh/square_hole_refined.png)
:-------------------:|:------------------:|:----:|

```python
import dmsh

geo = dmsh.Difference(dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0))
X, cells = dmsh.generate(geo, 0.1)
```
```python
import dmsh

geo = dmsh.Difference(
    dmsh.Circle([0.0, 0.0], 1.0),
    dmsh.Polygon([[0.0, 0.0], [1.5, 0.4], [1.5, -0.4]]),
)
X, cells = dmsh.generate(geo, 0.1, tol=1.0e-10)
```

The following example uses a nonconstant edge length; it depends on the distance to the
circle `c`.
```python
import dmsh
import numpy

r = dmsh.Rectangle(-1.0, +1.0, -1.0, +1.0)
c = dmsh.Circle([0.0, 0.0], 0.3)
geo = dmsh.Difference(r, c)

X, cells = dmsh.generate(
    geo, lambda pts: numpy.abs(c.dist(pts)) / 5 + 0.05, tol=1.0e-10
)
```

##### Union

![union](https://nschloe.github.io/dmsh/union.png) | ![union-rect](https://nschloe.github.io/dmsh/union_rectangles.png) | ![union-three-circles](https://nschloe.github.io/dmsh/union_three_circles.png) |
:-------------------:|:------------------:|:----:|

```python
import dmsh

geo = dmsh.Union([dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0)])
X, cells = dmsh.generate(geo, 0.15)
```
```python
import dmsh

geo = dmsh.Union(
    [dmsh.Rectangle(-1.0, +0.5, -1.0, +0.5), dmsh.Rectangle(-0.5, +1.0, -0.5, +1.0)]
)
X, cells = dmsh.generate(geo, 0.15)
```
```python
import dmsh
import numpy

angles = numpy.pi * numpy.array([3.0 / 6.0, 7.0 / 6.0, 11.0 / 6.0])
geo = dmsh.Union(
    [
        dmsh.Circle([numpy.cos(angles[0]), numpy.sin(angles[0])], 1.0),
        dmsh.Circle([numpy.cos(angles[1]), numpy.sin(angles[1])], 1.0),
        dmsh.Circle([numpy.cos(angles[2]), numpy.sin(angles[2])], 1.0),
    ]
)
X, cells = dmsh.generate(geo, 0.15)
```

#### Intersection

![intersection](https://nschloe.github.io/dmsh/intersection.png) | ![intersection-three-circles](https://nschloe.github.io/dmsh/intersection_three_circles.png) | ![halfspace](https://nschloe.github.io/dmsh/halfspace.png)
:-------------------:|:------------------:|:----:|

```python
import dmsh

geo = dmsh.Intersection([dmsh.Circle([0.0, -0.5], 1.0), dmsh.Circle([0.0, +0.5], 1.0)])
X, cells = dmsh.generate(geo, 0.1, tol=1.0e-10)
```

```python
import dmsh
import numpy

angles = numpy.pi * numpy.array([3.0 / 6.0, 7.0 / 6.0, 11.0 / 6.0])
geo = dmsh.Intersection(
    [
        dmsh.Circle([numpy.cos(angles[0]), numpy.sin(angles[0])], 1.5),
        dmsh.Circle([numpy.cos(angles[1]), numpy.sin(angles[1])], 1.5),
        dmsh.Circle([numpy.cos(angles[2]), numpy.sin(angles[2])], 1.5),
    ]
)
X, cells = dmsh.generate(geo, 0.1, tol=1.0e-10)
```

The following uses the `HalfSpace` primtive for cutting of a circle.
```python
import dmsh
import numpy

geo = dmsh.Intersection(
    [
        dmsh.HalfSpace(numpy.sqrt(0.5) * numpy.array([1.0, 1.0]), 0.0),
        dmsh.Circle([0.0, 0.0], 1.0),
    ]
)
X, cells = dmsh.generate(geo, 0.1)
```

### Rotation, translation, scaling

![rotation](https://nschloe.github.io/dmsh/rotation.png) | ![scaling](https://nschloe.github.io/dmsh/scaling.png)
|:----:|:----:|

```python
import dmsh
import numpy

geo = dmsh.Rotation(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), 0.1 * numpy.pi)
X, cells = dmsh.generate(geo, 0.1, tol=1.0e-10)
```
```python
import dmsh

geo = dmsh.Translation(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), [1.0, 1.0])
X, cells = dmsh.generate(geo, 0.1)
```
```python
import dmsh

geo = dmsh.Scaling(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), 2.0)
X, cells = dmsh.generate(geo, 0.1, tol=1.0e-5)
```

### Local refinement

![refinement-line](https://nschloe.github.io/dmsh/refinement_line.png)

All objects can be used to refine the mesh according to the distance to the object;
e.g. a `Path`:
```python
import dmsh

geo = dmsh.Rectangle(0.0, 1.0, 0.0, 1.0)

p1 = dmsh.Path([[0.4, 0.6], [0.6, 0.4]])


def edge_size(x):
    return 0.03 + 0.1 * p1.dist(x)


X, cells = dmsh.generate(geo, edge_size, tol=1.0e-10)
```


### Custom shapes
It is also possible to define your own geometry. Simply create a class derived from
`dmsh.Geometry` that contains a `dist` method and a method to project points onto the
boundary.

```python
import dmsh
import numpy


class MyDisk(dmsh.Geometry):
    def __init__(self):
        super().__init__()
        self.r = 1.0
        self.x0 = [0.0, 0.0]
        self.bounding_box = [-1.0, 1.0, -1.0, 1.0]
        self.feature_points = numpy.array([[], []]).T

    def dist(self, x):
        assert x.shape[0] == 2
        y = (x.T - self.x0).T
        return numpy.sqrt(numpy.einsum("i...,i...->...", y, y)) - self.r

    def boundary_step(self, x):
        # project onto the circle
        y = (x.T - self.x0).T
        r = numpy.sqrt(numpy.einsum("ij,ij->j", y, y))
        return ((y / r * self.r).T + self.x0).T


geo = MyDisk()
X, cells = dmsh.generate(geo, 0.1)
```

### Debugging

![level-set-poly](https://nschloe.github.io/dmsh/levelset-polygon.png) | ![level-set-rect-hole](https://nschloe.github.io/dmsh/levelset-rect-hole.png)
|:----:|:----:|

dmsh is rather fragile, but sometimes the break-downs are due to an incorrectly defined
geometry. Use
```
geo.show()
```
to inspect the level set function of your domain. (It must be negative inside the
domain and positive outside. The 0-level set forms the domain boundary.)


### Installation

dmsh is [available from the Python Package
Index](https://pypi.org/project/dmsh/), so simply type
```
pip install dmsh
```
to install.

### Testing

To run the dmsh unit tests, check out this repository and type
```
MPLBACKEND=Agg pytest
```
(Setting the environment variable prevents the test figures from being displayed.)

### License
This software is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
