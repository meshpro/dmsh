from __future__ import print_function

from .__about__ import __author__, __email__, __license__, __status__, __version__
from .geometry import (
    Circle,
    Difference,
    Ellipse,
    HalfSpace,
    Intersection,
    Path,
    Polygon,
    Rectangle,
    Rotation,
    Scaling,
    Stretch,
    Translation,
    Union,
)
from .main import generate

__all__ = [
    "__author__",
    "__email__",
    "__license__",
    "__version__",
    "__status__",
    "generate",
    "Circle",
    "Difference",
    "Ellipse",
    "HalfSpace",
    "Intersection",
    "Path",
    "Polygon",
    "Rectangle",
    "Rotation",
    "Stretch",
    "Scaling",
    "Translation",
    "Union",
]

# try:
#     import pipdate
# except ImportError:
#     pass
# else:
#     if pipdate.needs_checking(__name__):
#         print(pipdate.check(__name__, __version__), end="")
