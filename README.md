# dmsh

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/dmsh/master.svg)](https://circleci.com/gh/nschloe/dmsh/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/dmsh.svg)](https://codecov.io/gh/nschloe/dmsh)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![PyPi Version](https://img.shields.io/pypi/v/dmsh.svg)](https://pypi.org/project/dmsh)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/dmsh.svg?logo=github&label=Stars)](https://github.com/nschloe/dmsh)

The worst mesh generator you'll ever use.

Inspired by [distmesh](http://persson.berkeley.edu/distmesh/), dmsh is slow, requires a
lot of memory, and isn't terribly robust either. It's got a simple enough interface
though, and if it works, it produces pretty nice meshes. Might be useful.

### Examples

#### Primitives

![circle](https://nschloe.github.io/dmsh/circle.png) |
![rectangle](https://nschloe.github.io/dmsh/rectangle.png) |
![polygon](https://nschloe.github.io/dmsh/polygon.png)
|:---:|:---:|:---:|

```python
import dmsh

geo = dmsh.Circle([0.0, 0.0], 1.0)
X, cells = dmsh.generate(geo, 0.1)

# import meshio
# meshio.write_points_and_cells("circle.vtk", X, {"triangle": cells})
```

```python
geo = dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0)
X, cells = dmsh.generate(geo, 0.1)
```

```python
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

![difference](https://nschloe.github.io/dmsh/difference.png) |
![pacman](https://nschloe.github.io/dmsh/pacman.png) |
![square_hole_refined](https://nschloe.github.io/dmsh/square_hole_refined.png)
:-------------------:|:------------------:|:----:|

```python
geo = dmsh.Difference(dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0))
X, cells = dmsh.generate(geo, 0.1)
```
```python
geo = dmsh.Difference(
    dmsh.Circle([0.0, 0.0], 1.0),
    dmsh.Polygon([[0.0, 0.0], [1.5, 0.4], [1.5, -0.4]]),
)
X, cells = dmsh.generate(geo, 0.1, tol=1.0e-10)
```

The following example uses a nonconstant edge length; it depends on the distance to the
circle `c`.
```python
r = dmsh.Rectangle(-1.0, +1.0, -1.0, +1.0)
c = dmsh.Circle([0.0, 0.0], 0.3)
geo = dmsh.Difference(r, c)

numpy.random.seed(0)
X, cells = dmsh.generate(
    geo, lambda pts: numpy.abs(c.dist(pts)) / 5 + 0.05, tol=1.0e-10
)
```

##### Union

![union](https://nschloe.github.io/dmsh/union.png) |
![union-rect](https://nschloe.github.io/dmsh/union_rectangles.png) |
![union-three-circles](https://nschloe.github.io/dmsh/union_three_circles.png) |
:-------------------:|:------------------:|:----:|

```python
geo = dmsh.Union([dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0)])
X, cells = dmsh.generate(geo, 0.15)
```
```python
geo = dmsh.Union(
    [dmsh.Rectangle(-1.0, +0.5, -1.0, +0.5), dmsh.Rectangle(-0.5, +1.0, -0.5, +1.0)]
)
X, cells = dmsh.generate(geo, 0.15)
```
```python
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

![intersection](https://nschloe.github.io/dmsh/intersection.png) |
![intersection-three-circles](https://nschloe.github.io/dmsh/intersection_three_circles.png) |
![halfspace](https://nschloe.github.io/dmsh/halfspace.png)
:-------------------:|:------------------:|:----:|

```python
geo = dmsh.Intersection(
    [dmsh.Circle([0.0, -0.5], 1.0), dmsh.Circle([0.0, +0.5], 1.0)]
)
X, cells = dmsh.generate(geo, 0.1, tol=1.0e-10)
```

```python
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
geo = dmsh.Intersection(
    [
        dmsh.HalfSpace(numpy.sqrt(0.5) * numpy.array([1.0, 1.0]), 0.0),
        dmsh.Circle([0.0, 0.0], 1.0),
    ]
)
X, cells = dmsh.generate(geo, 0.1)
```

### Rotation, translation, scaling

![rotation](https://nschloe.github.io/dmsh/rotation.png) |
![scaling](https://nschloe.github.io/dmsh/scaling.png)
|:----:|:----:|

```python
geo = dmsh.Rotation(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), 0.1 * numpy.pi)
X, cells = dmsh.generate(geo, 0.1, tol=1.0e-10)
```
```python
geo = dmsh.Translation(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), [1.0, 1.0])
X, cells = dmsh.generate(geo, 0.1, show=show)
```
```python
geo = dmsh.Scaling(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), 2.0)
X, cells = dmsh.generate(geo, 0.1, show=show, tol=1.0e-5)
```

### Local refinement

![refinement-line](https://nschloe.github.io/dmsh/refinement_line.png)

All objects can be used to refine the mesh according to the distance to the object;
e.g. a `Path`:
```
geo = dmsh.Rectangle(0.0, 1.0, 0.0, 1.0)

p1 = dmsh.Path([[0.4, 0.6], [0.6, 0.4]])

def edge_size(x):
    return 0.03 + 0.1 * p1.dist(x)

X, cells = dmsh.generate(geo, edge_size, show=show, tol=1.0e-10)
```



### Installation

dmsh is [available from the Python Package
Index](https://pypi.org/project/dmsh/), so simply type
```
pip install -U dmsh
```
to install or upgrade.

### Testing

To run the dmsh unit tests, check out this repository and type
```
pytest
```

### Distribution

To create a new release

1. bump the `__version__` number,

2. publish to PyPi and GitHub:
    ```
    make publish
    ```

### License

dmsh is published under the [MIT license](https://en.wikipedia.org/wiki/MIT_License).
