# -*- coding: utf-8 -*-
#
import numpy


def assert_norm_equality(X, ref_norm, tol):
    val = numpy.linalg.norm(X, ord=1)
    assert (
        abs(val - ref_norm[0]) < tol * ref_norm[0]
    ), "Expected: {:.16e}   Computed: {:.16e}".format(ref_norm[0], val)

    val = numpy.linalg.norm(X, ord=2)
    assert (
        abs(val - ref_norm[1]) < tol * ref_norm[1]
    ), "Expected: {:.16e}   Computed: {:.16e}".format(ref_norm[1], val)

    val = numpy.linalg.norm(X, ord=numpy.inf)
    assert (
        abs(val - ref_norm[2]) < tol * ref_norm[2]
    ), "Expected: {:.16e}   Computed: {:.16e}".format(ref_norm[2], val)
    return
