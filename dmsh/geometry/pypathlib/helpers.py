import numpy as np


def shoelace(x):
    previous = np.roll(x, 1, axis=0)
    return np.sum(x[..., 1] * previous[..., 0] - x[..., 0] * previous[..., 1], axis=0)
