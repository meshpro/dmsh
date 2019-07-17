import meshio
import meshzoo

points, cells = meshzoo.triangle(2)
meshio.write_points_cells("logo.svg", points, {"triangle": cells})
