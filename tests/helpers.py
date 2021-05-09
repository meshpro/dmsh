import numpy as np


def assert_equality(a, b, tol):
    a = np.asarray(a)
    b = np.asarray(b)
    fmt_a = ", ".join(["{:.16e}"] * len(a))
    fmt_b = ", ".join(["{:.16e}"] * len(b))
    assert np.all(np.abs(a - b) < tol), f"[{fmt_a}]\n[{fmt_b}]".format(*a, *b)


def assert_norm_equality(X, ref_norm, tol):
    ref_norm = np.asarray(ref_norm)
    vals = np.array(
        [
            np.linalg.norm(X, ord=1),
            np.linalg.norm(X, ord=2),
            np.linalg.norm(X, ord=np.inf),
        ]
    )
    assert np.all(
        np.abs(vals - ref_norm) < tol * ref_norm
    ), "Expected: [{:.16e}, {:.16e}, {:.16e}]\nComputed: [{:.16e}, {:.16e}, {:.16e}]".format(
        *ref_norm, *vals
    )


def save(filename, X, cells):
    import meshplex

    mesh = meshplex.MeshTri(X, cells)
    mesh.save(
        filename,
        show_coedges=False,
        show_axes=False,
        nondelaunay_edge_color="k",
    )
