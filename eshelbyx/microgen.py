import pyvista as pv
import numpy as np


def voxelize_on_grid(mesh: pv.PolyData, grid: pv.ImageData):
    """
    Project a surface mesh onto an existing grid.
    Returns:
        inside_cells : (nx-1, ny-1, nz-1) boolean array
    """
    centers = grid.cell_centers()

    enclosed = centers.select_enclosed_points(mesh, tolerance=0.0)
    inside = enclosed["SelectedPoints"].view(np.bool_)

    nx, ny, nz = grid.dimensions
    inside_cells = inside.reshape((nx - 1, ny - 1, nz - 1), order="F")

    return inside_cells


def make_grid(bounds, dims):
    nx, ny, nz = dims
    xmin, xmax, ymin, ymax, zmin, zmax = bounds

    grid = pv.ImageData()
    grid.dimensions = (nx, ny, nz)
    grid.origin = (xmin, ymin, zmin)
    grid.spacing = (
        (xmax - xmin) / (nx - 1),
        (ymax - ymin) / (ny - 1),
        (zmax - zmin) / (nz - 1),
    )
    return grid


def main():
    N = 32
    radius = 0.25

    bounds = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    grid = make_grid(bounds, (N + 1, N + 1, N + 1))
    box = pv.Box(bounds=bounds)

    # Phase 1: pore
    pore = pv.Sphere(radius=radius, center=(0, 0, 0)).triangulate()

    # Phase 2: inclusions
    spheres = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                spheres.append(
                    pv.Sphere(radius=radius, center=(i * 0.5, j * 0.5, k * 0.5))
                )

    def merge_primitives(prims):
        m = prims[0]
        for p in prims[1:]:
            m = m.merge(p)
        return m.clean().triangulate()

    inclusions = merge_primitives(spheres)

    # ------------------
    # Voxelize all phases
    # ------------------
    inside_pore = voxelize_on_grid(pore, grid)
    inside_inclusions = voxelize_on_grid(inclusions, grid)

    # ------------------
    # Phase labeling
    # ------------------
    # 0 = matrix, 1 = pore, 2 = inclusion
    phase = np.zeros_like(inside_pore, dtype=np.uint8)
    phase[inside_pore] = 1
    phase[inside_inclusions] = 2

    grid.cell_data["phase"] = phase.flatten(order="F")

    print("Volume fractions:")
    print("matrix    :", np.mean(phase == 0))
    print("pore      :", np.mean(phase == 1))
    print("inclusion :", np.mean(phase == 2))

    # ------------------
    # Visualization
    # ------------------
    plotter = pv.Plotter(shape=(1, 3))

    plotter.subplot(0, 0)
    plotter.add_mesh(box, color="lightblue", opacity=0.2)
    plotter.add_mesh(pore, color="black", opacity=0.6)
    plotter.add_mesh(inclusions, color="red", opacity=0.7)
    plotter.add_title("Continuous geometry")

    plotter.subplot(0, 1)
    vox_inc = grid.threshold(1.5, scalars="phase")
    plotter.add_mesh(vox_inc, show_edges=True)
    vox_pore = grid.threshold([0.5, 1.5], scalars="phase")
    plotter.add_mesh(vox_pore, show_edges=True)
    plotter.add_title("Voxelized inclusions")

    plotter.subplot(0, 2)
    matrix = grid.threshold(0.5, scalars="phase", invert=True)
    plotter.add_mesh(matrix, show_edges=True, opacity=0.4, scalars=None)
    plotter.add_title("Voxelized matrix")
    plotter.show()

    grid.save(f"3_phase_microstructure_{N}.vti")
    return grid, phase


main()
