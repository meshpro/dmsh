# -*- coding: utf-8 -*-
#
import numpy


def assert_norm_equality(X, ref_norm, tol):
    ref_norm = numpy.array(ref_norm)
    vals = numpy.array(
        [
            numpy.linalg.norm(X, ord=1),
            numpy.linalg.norm(X, ord=2),
            numpy.linalg.norm(X, ord=numpy.inf),
        ]
    )
    assert numpy.all(
        numpy.abs(vals - ref_norm) < tol * ref_norm
    ), "Expected: [{:.16e}, {:.16e}, {:.16e}]\nComputed: [{:.16e}, {:.16e}, {:.16e}]".format(
        *ref_norm, *vals
    )
    return
