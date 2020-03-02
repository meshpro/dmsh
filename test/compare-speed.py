import time

import matplotlib.pyplot as plt
import numpy as np

import dmsh
import meshplex
import optimesh
import pygmsh


def _compute_num_boundary_points(total_num_points):
    # The number of boundary points, the total number of points, and the number of cells
    # are connected by two equations (the second of which is approximate).
    #
    # Euler:
    # 2 * num_points - num_boundary_edges - 2 = num_cells
    #
    # edge_length = 2 * np.pi / num_boundary_points
    # tri_area = np.sqrt(3) / 4 * edge_length ** 2
    # num_cells = int(np.pi / tri_area)
    #
    # num_boundary_points = num_boundary_edges
    #
    # Hence:
    # 2 * num_points =
    # num_boundary_points + 2 + np.pi / (np.sqrt(3) / 4 * (2 * np.pi / num_boundary_points) ** 2)
    #
    # We need to solve
    #
    # + num_boundary_points ** 2
    # + (sqrt(3) * pi) * num_boundary_points
    # + (2 - 2 * num_points) * (sqrt(3) * pi)
    # = 0
    #
    # for the number of boundary points.
    sqrt3_pi = np.sqrt(3) * np.pi
    num_boundary_points = -sqrt3_pi / 2 + np.sqrt(
        3 / 4 * np.pi ** 2 - (2 - 2 * total_num_points) * sqrt3_pi
    )
    return num_boundary_points


def dmsh_circle(num_points):
    target_edge_length = 2 * np.pi / _compute_num_boundary_points(num_points)
    geo = dmsh.Circle([0.0, 0.0], 1.0)
    X, cells = dmsh.generate(geo, target_edge_length)
    return X, cells


def gmsh_circle(num_points):
    geom = pygmsh.built_in.Geometry()
    target_edge_length = 2 * np.pi / _compute_num_boundary_points(num_points)
    geom.add_circle(
        [0.0, 0.0, 0.0], 1.0, lcar=target_edge_length, num_sections=4, compound=True
    )
    mesh = pygmsh.generate_mesh(geom, remove_lower_dim_cells=True, verbose=False)
    return mesh.points[:, :2], mesh.cells[0].data


data = {
    "dmsh": {"n": [], "time": [], "q": []},
    "gmsh": {"n": [], "time": [], "q": []},
}
for num_points in range(1000, 10000, 1000):
    print(num_points)
    # dmsh
    t = time.time()
    pts, cells = dmsh_circle(num_points)
    t = time.time() - t
    mesh = meshplex.MeshTri(pts, cells)
    avg_q = np.sum(mesh.cell_quality) / len(mesh.cell_quality)
    data["dmsh"]["n"].append(len(pts))
    data["dmsh"]["time"].append(t)
    data["dmsh"]["q"].append(avg_q)

    # gmsh
    t = time.time()
    pts, cells = gmsh_circle(num_points)
    t = time.time() - t
    mesh = meshplex.MeshTri(pts, cells)
    avg_q = np.sum(mesh.cell_quality) / len(mesh.cell_quality)
    data["gmsh"]["n"].append(len(pts))
    data["gmsh"]["time"].append(t)
    data["gmsh"]["q"].append(avg_q)


# plot condition number
for key, value in data.items():
    plt.plot(value["n"], value["time"], "-x", label=key)
plt.xlabel("num points")
plt.title("generation time [s]")
plt.grid()
plt.legend()
# plt.show()
plt.savefig("time.svg", transparent=True, bbox_inches="tight")
plt.close()

# plot CG iterations number
for key, value in data.items():
    plt.plot(value["n"], value["q"], "-x", label=key)
plt.xlabel("num points")
plt.title("average cell quality")
plt.grid()
plt.legend()
# plt.show()
plt.savefig("average-cell-quality.svg", transparent=True, bbox_inches="tight")
plt.close()
