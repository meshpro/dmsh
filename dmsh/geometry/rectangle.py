# -*- coding: utf-8 -*-
#
import numpy

from .polygon import Polygon


class Rectangle(Polygon):
    def __init__(self, x0, x1, y0, y1):
        points = numpy.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
        super(Rectangle, self).__init__(points)
        return
