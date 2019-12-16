import numpy


def shoelace(x):
    previous = numpy.roll(x, 1, axis=0)
    return numpy.sum(
        x[..., 1] * previous[..., 0] - x[..., 0] * previous[..., 1], axis=0
    )
