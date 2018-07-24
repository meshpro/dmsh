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

##### Circle

![circle](https://nschloe.github.io/dmsh/circle.png)

```python
import dmsh

geo = dmsh.Circle([0.0, 0.0], 1.0)
X, cells = dmsh.generate(geo, 0.1)

# import meshio
# meshio.write_points_and_cells("circle.vtk", X, {"triangle": cells})
```

##### Rectangle

![rectangle](https://nschloe.github.io/dmsh/rectangle.png)

```python
geo = dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0)
X, cells = dmsh.generate(geo, 0.1)
```

##### Polygon

![polygon](https://nschloe.github.io/dmsh/polygon.png)

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

##### Union

![union](https://nschloe.github.io/dmsh/union.png) |
![union-rect](https://nschloe.github.io/dmsh/union_rectangles.png) |
:-------------------:|:------------------:|

```python
geo = dmsh.Union([dmsh.Circle([-0.5, 0.0], 1.0), dmsh.Circle([+0.5, 0.0], 1.0)])
# geo = dmsh.Union(
#     [dmsh.Rectangle(-1.0, +0.5, -1.0, +0.5), dmsh.Rectangle(-0.5, +1.0, -0.5, +1.0)]
# )
X, cells = dmsh.generate(geo, 0.15)
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
