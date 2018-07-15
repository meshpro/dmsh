# -*- coding: utf-8 -*-
#
import dmsh
import meshio


def test_dmsh():
    X, cells = dmsh.generate()
    print(X)
    print(cells)
    meshio.write_points_cells(X, cells)
    return


if __name__ == "__main__":
    test_dmsh()
