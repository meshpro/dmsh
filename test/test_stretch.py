# import dmsh
#
# from helpers import assert_norm_equality, save
#
#
# def test(show=True):
#     geo = dmsh.Stretch(dmsh.Rectangle(-1.0, +2.0, -1.0, +1.0), [1.0, 1.0])
#     X, cells = dmsh.generate(geo, 0.2, show=show, tol=1.0e-3)
#
#     ref_norms = [4.3336958624757887e+02, 2.3620906390718797e+01, 2.6213203435596428e+00]
#     assert_norm_equality(X.flatten(), ref_norms, 1.0e-12)
#     return X, cells
#
#
# if __name__ == "__main__":
#     X, cells = test(show=False)
#     save("stretch.png", X, cells)
