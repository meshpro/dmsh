# dmsh

[![CircleCI](https://img.shields.io/circleci/project/github/nschloe/dmsh/master.svg)](https://circleci.com/gh/nschloe/dmsh/tree/master)
[![codecov](https://img.shields.io/codecov/c/github/nschloe/dmsh.svg)](https://codecov.io/gh/nschloe/dmsh)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)
[![PyPi Version](https://img.shields.io/pypi/v/dmsh.svg)](https://pypi.org/project/dmsh)
[![GitHub stars](https://img.shields.io/github/stars/nschloe/dmsh.svg?logo=github&label=Stars)](https://github.com/nschloe/dmsh)

MEsh generator inspired by [distmesh](http://persson.berkeley.edu/distmesh/).


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
