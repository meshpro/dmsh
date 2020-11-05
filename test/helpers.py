import numpy


def assert_equality(a, b, tol):
    """
    Compares two arrays.

    Args:
        a: (todo): write your description
        b: (todo): write your description
        tol: (float): write your description
    """
    a = numpy.asarray(a)
    b = numpy.asarray(b)
    fmt_a = ", ".join(["{:.16e}"] * len(a))
    fmt_b = ", ".join(["{:.16e}"] * len(b))
    assert numpy.all(numpy.abs(a - b) < tol), f"[{fmt_a}]\n[{fmt_b}]".format(*a, *b)


def assert_norm_equality(X, ref_norm, tol):
    """
    Asserts the norm_norm.

    Args:
        X: (todo): write your description
        ref_norm: (todo): write your description
        tol: (float): write your description
    """
    ref_norm = numpy.asarray(ref_norm)
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


def save(filename, X, cells):
    """
    Save a mesh as an svg file.

    Args:
        filename: (str): write your description
        X: (dict): write your description
        cells: (dict): write your description
    """
    import meshplex

    mesh = meshplex.MeshTri(X, cells)
    mesh.save(
        filename,
        show_coedges=False,
        show_axes=False,
        nondelaunay_edge_color="k",
    )
