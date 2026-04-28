import pyvista as pv
import numpy as np
import jax.numpy as jnp
# from jaxmat.tensors import SymmetricTensor2, IsotropicTensor4, SymmetricTensor4


def voxelize_on_grid(mesh: pv.PolyData, grid: pv.ImageData):
    """
    Project a surface mesh onto an existing grid.
    Returns:
        inside_cells : (nx-1, ny-1, nz-1) boolean array
    """
    centers = grid.cell_centers()

    enclosed = centers.select_enclosed_points(mesh, tolerance=1e-16)
    inside = enclosed["SelectedPoints"].view(np.bool_)

    nx, ny, nz = grid.dimensions
    inside_cells = inside.reshape((nx - 1, ny - 1, nz - 1), order="F")

    return inside_cells


def voxelize_all_corners(mesh: pv.PolyData, grid: pv.ImageData):
    """
    Function similar to "voxelize_on_grid" but boolean returned are True if all corners of the voxel
    are in the mesh (and not juste the center as in "voxelise_on_grid").
    """
    nx, ny, nz = grid.dimensions

    pts = pv.PolyData(grid.points)
    mask = pts.select_enclosed_points(mesh)["SelectedPoints"].astype(bool)

    inside = mask.reshape(nx, ny, nz)

    voxels = (
        inside[:-1, :-1, :-1]
        & inside[1:, :-1, :-1]
        & inside[:-1, 1:, :-1]
        & inside[1:, 1:, :-1]
        & inside[:-1, :-1, 1:]
        & inside[1:, :-1, 1:]
        & inside[:-1, 1:, 1:]
        & inside[1:, 1:, 1:]
    )
    return voxels


def voxelize_at_least_one_corner(mesh: pv.PolyData, grid: pv.ImageData):
    """
    Function similar to "voxelize_on_grid" but boolean returned are True at least one corner of the voxel
    is in the mesh (and not juste the center as in "voxelise_on_grid").
    """

    nx, ny, nz = grid.dimensions

    pts = pv.PolyData(grid.points)
    mask = pts.select_enclosed_points(mesh)["SelectedPoints"].astype(bool)

    inside = mask.reshape(nx, ny, nz)

    voxels = (
        inside[:-1, :-1, :-1]
        | inside[1:, :-1, :-1]
        | inside[:-1, 1:, :-1]
        | inside[1:, 1:, :-1]
        | inside[:-1, :-1, 1:]
        | inside[1:, :-1, 1:]
        | inside[:-1, 1:, 1:]
        | inside[1:, 1:, 1:]
    )
    return voxels


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


def main_3_phases():
    """
    Voxelized geometry of a matrix cube with a circular pore in the center, 8 inclusions in the corners
    (without hybrid voxels).
    """

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


def main_Hashin_an():
    """
    Analytical solution for the Hashin's coated sphere
    """
    N = 16
    Nloc = 5
    Rint = 0.2
    Rext = 0.4
    coating_thickness = Rext - Rint

    # Material parameters from Matti Schneider's paper : M. Schneider et al. Computational homogenization of
    # elasticity on a staggered grid, International Journal for Numerical Methods in Engineering, 105, 693–720. (2016)
    phi1 = (Rint / Rext) ** 3
    phi2 = 1 - phi1
    soft_center = True  # True : pore softer than the coating
    chi = 1000
    Keff = 1.0
    mueff = 0.6

    if soft_center == True:
        K1 = (9 + (5 * ((1 / chi) - 1) * phi2)) * Keff / (9 - (4 * (1 - chi) * phi2))
        K2 = K1 * chi
    else:
        K1 = (9 - (5 * (1 - chi) * phi2)) * Keff / (9 + (4 * ((1 / chi) - 1) * phi2))
        K2 = K1 / chi

    mu1 = 3 * K1 / 5
    mu2 = 3 * K2 / 5

    aeff = 1.0
    a2 = (3 * Keff + 4 * mu2) / (3 * K2 + 4 * mu2)
    b2 = (3 * ((Rint + coating_thickness) ** 3) * (K2 - Keff)) / (3 * K2 + 4 * mu2)
    a1 = a2 + (b2 / (Rint**3))

    bounds = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    grid_coarse = make_grid(bounds, (N + 1, N + 1, N + 1))

    pore = pv.Sphere(
        radius=Rint, center=(0, 0, 0)
    ).triangulate()  # Pore geometry in Pyvista
    coating_outer = pv.Sphere(
        radius=Rint + coating_thickness, center=(0, 0, 0)
    ).triangulate()  # Pore + coating geometry in Pyvista

    # ------------------
    # Voxelize all phases
    # ------------------
    inside_pore = voxelize_on_grid(pore, grid_coarse)  # Voxelized pore
    inside_outer = voxelize_on_grid(coating_outer, grid_coarse)
    inside_coating = inside_outer & (~inside_pore)  # Voxelized coating

    # --------------------------------------------------------
    # Phase coarse analytique
    # --------------------------------------------------------
    phase_coarse = np.zeros((N, N, N), dtype=np.uint8)

    # Coordonnées des centres des voxels
    dx = (bounds[1] - bounds[0]) / N
    xc = bounds[0] + (np.arange(N) + 0.5) * dx
    yc = bounds[2] + (np.arange(N) + 0.5) * dx
    zc = bounds[4] + (np.arange(N) + 0.5) * dx

    Xc, Yc, Zc = np.meshgrid(xc, yc, zc, indexing="ij")
    rc = np.sqrt(Xc**2 + Yc**2 + Zc**2)

    phase_coarse[rc <= Rint] = 1  # Pore
    phase_coarse[(rc > Rint) & (rc <= Rext)] = 2  # Coating
    phase_coarse[rc > Rext] = 0  # Matrix

    grid_coarse.cell_data["phase"] = phase_coarse.flatten()

    # --------------------
    # Grille coarse directe
    # --------------------
    dx = (bounds[1] - bounds[0]) / N

    # Stockage résultat
    sigma_coarse = np.zeros((N, N, N, 3, 3))

    eps_coarse = np.zeros((N, N, N, 3, 3))

    W_elast_coarse = np.zeros((N, N, N))

    # Boucle sur voxels coarse
    for i in range(N):
        for j in range(N):
            for k in range(N):
                # Coordonnées du voxel coarse (coin min)
                x0 = bounds[0] + i * dx
                y0 = bounds[2] + j * dx
                z0 = bounds[4] + k * dx

                # Sous-grille locale
                xs = np.linspace(x0, x0 + dx, Nloc)
                ys = np.linspace(y0, y0 + dx, Nloc)
                zs = np.linspace(z0, z0 + dx, Nloc)

                X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")

                r = np.sqrt(X**2 + Y**2 + Z**2)
                r_safe = np.maximum(r, 1e-12)

                # Phase locale analytique (évite voxelisation !)
                phase_loc = np.zeros_like(r, dtype=np.uint8)
                phase_loc[r <= Rint] = 1
                phase_loc[(r > Rint) & (r <= Rext)] = 2

                # Contraintes sphériques
                sigma_rr = np.zeros_like(r)
                sigma_tt = np.zeros_like(r)
                sigma_pp = np.zeros_like(r)

                eps_rr = np.zeros_like(r)
                eps_tt = np.zeros_like(r)
                eps_pp = np.zeros_like(r)

                sigma_rr[phase_loc == 1] = 3 * K1 * a1
                sigma_tt[phase_loc == 1] = 3 * K1 * a1
                sigma_pp[phase_loc == 1] = 3 * K1 * a1

                eps_rr[phase_loc == 1] = a1
                eps_tt[phase_loc == 1] = a1
                eps_pp[phase_loc == 1] = a1

                mask2 = phase_loc == 2
                sigma_rr[mask2] = 3 * K2 * a2 - (4 * mu2 * b2 / r_safe[mask2] ** 3)
                sigma_tt[mask2] = 3 * K2 * a2 + (2 * mu2 * b2 / r_safe[mask2] ** 3)
                sigma_pp[mask2] = 3 * K2 * a2 + (2 * mu2 * b2 / r_safe[mask2] ** 3)

                eps_rr[mask2] = a2 - (2 * b2 / r_safe[mask2] ** 3)
                eps_tt[mask2] = a2 + (b2 / r_safe[mask2] ** 3)
                eps_pp[mask2] = a2 + (b2 / r_safe[mask2] ** 3)

                sigma_rr[phase_loc == 0] = 3 * Keff * aeff
                sigma_tt[phase_loc == 0] = 3 * Keff * aeff
                sigma_pp[phase_loc == 0] = 3 * Keff * aeff

                eps_rr[phase_loc == 0] = aeff
                eps_tt[phase_loc == 0] = aeff
                eps_pp[phase_loc == 0] = aeff

                # Passage sphérique → cartésien
                theta = np.arccos(Z / r_safe)
                phi = np.atan2(Y, X)

                st = np.sin(theta)
                ct = np.cos(theta)
                sp = np.sin(phi)
                cp = np.cos(phi)

                er = np.stack((st * cp, st * sp, ct), axis=-1)
                et = np.stack((ct * cp, ct * sp, -st), axis=-1)
                ep = np.stack((-sp, cp, np.zeros_like(r)), axis=-1)

                sigma = (
                    sigma_rr[..., None, None] * er[..., :, None] * er[..., None, :]
                    + sigma_tt[..., None, None] * et[..., :, None] * et[..., None, :]
                    + sigma_pp[..., None, None] * ep[..., :, None] * ep[..., None, :]
                )

                eps = (
                    eps_rr[..., None, None] * er[..., :, None] * er[..., None, :]
                    + eps_tt[..., None, None] * et[..., :, None] * et[..., None, :]
                    + eps_pp[..., None, None] * ep[..., :, None] * ep[..., None, :]
                )

                W_elast = 0.5 * np.einsum("...ij,...ij->...", sigma, eps)

                # Moyenne locale → voxel coarse
                sigma_coarse[i, j, k] = sigma.mean(axis=(0, 1, 2))
                eps_coarse[i, j, k] = eps.mean(axis=(0, 1, 2))
                W_elast_coarse[i, j, k] = W_elast.mean()

    sigma_xx = sigma_coarse[..., 0, 0]
    sigma_yy = sigma_coarse[..., 1, 1]
    sigma_zz = sigma_coarse[..., 2, 2]
    sigma_xy = sigma_coarse[..., 0, 1]
    sigma_xz = sigma_coarse[..., 0, 2]
    sigma_yz = sigma_coarse[..., 1, 2]

    eps_xx = eps_coarse[..., 0, 0]
    eps_yy = eps_coarse[..., 1, 1]
    eps_zz = eps_coarse[..., 2, 2]
    eps_xy = eps_coarse[..., 0, 1]
    eps_xz = eps_coarse[..., 0, 2]
    eps_yz = eps_coarse[..., 1, 2]

    grid_coarse.cell_data["sigma_xx_an"] = sigma_xx.flatten()
    grid_coarse.cell_data["sigma_yy_an"] = sigma_yy.flatten()
    grid_coarse.cell_data["sigma_zz_an"] = sigma_zz.flatten()
    grid_coarse.cell_data["sigma_xy_an"] = sigma_xy.flatten()
    grid_coarse.cell_data["sigma_xz_an"] = sigma_xz.flatten()
    grid_coarse.cell_data["sigma_yz_an"] = sigma_yz.flatten()

    grid_coarse.cell_data["eps_xx_an"] = eps_xx.flatten()
    grid_coarse.cell_data["eps_yy_an"] = eps_yy.flatten()
    grid_coarse.cell_data["eps_zz_an"] = eps_zz.flatten()
    grid_coarse.cell_data["eps_xy_an"] = eps_xy.flatten()
    grid_coarse.cell_data["eps_xz_an"] = eps_xz.flatten()
    grid_coarse.cell_data["eps_yz_an"] = eps_yz.flatten()

    grid_coarse.cell_data["W_elast"] = W_elast_coarse.flatten()

    # ------------------
    # Visualization
    # ------------------
    print("Volume fractions:")
    print("matrix    :", np.mean(phase_coarse == 0))
    print("pore      :", np.mean(phase_coarse == 1))
    print("coating :", np.mean(phase_coarse == 2))

    plotter = pv.Plotter()

    # slice at z = 0
    slice_mid = grid_coarse.slice(normal="z", origin=(0, 0, 0))

    # Plot phases
    plotter.add_mesh(
        slice_mid, scalars="phase", show_edges=True, cmap=["lightblue", "black", "red"]
    )
    plotter.add_title("Transverse slice at z=0")
    plotter.show()

    grid_coarse.save(f"results/Hashin_an_{N}.vti")
    return grid_coarse, phase_coarse


def main_Hashin_num():
    """
    Voxelized geometry of the Hashin's coated sphere problem for future numerical computation
    """

    N = 16  # Number of voxels in each direction
    Rint = 0.2
    Rext = 0.4
    coating_thickness = Rext - Rint

    bounds = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)  # Boundaries of the domain
    grid = make_grid(bounds, (N + 1, N + 1, N + 1))

    pore = pv.Sphere(
        radius=Rint, center=(0, 0, 0)
    ).triangulate()  # Pore geometry in Pyvista
    coating_outer = pv.Sphere(
        radius=Rint + coating_thickness, center=(0, 0, 0)
    ).triangulate()  # Pore + coating geometry in Pyvista

    # ------------------
    # Voxelize all phases
    # ------------------
    inside_pore = voxelize_all_corners(pore, grid)
    all_pore = voxelize_at_least_one_corner(pore, grid)
    hybrid_pore = all_pore & (
        ~inside_pore
    )  # Voxelized hybrid voxels at the interface between central sphere and coating

    inside_coating = voxelize_all_corners(coating_outer, grid) & (~all_pore)
    all_coating_outer = voxelize_at_least_one_corner(coating_outer, grid)
    hybrid_coating = (
        all_coating_outer & (~inside_coating) & (~all_pore)
    )  # Voxelized hybrid voxels at the interface between coating and homogenized matrix

    inside_pore_bis = voxelize_on_grid(pore, grid)  # Voxelized central sphere
    inside_coating_bis = voxelize_on_grid(coating_outer, grid) & (
        ~inside_pore_bis
    )  # Voxelized coating

    # ------------------
    # Phase labeling
    # ------------------
    # 0 = matrix, 1 = pore, 2 = coating
    phase = np.zeros_like(inside_pore_bis, dtype=np.uint8)
    phase[inside_pore_bis] = 1
    phase[inside_coating_bis] = 2

    grid.cell_data["phase"] = phase.flatten()

    # -------------------------------------------
    # Hybrid voxel labeling (hybrid or not)
    # -------------------------------------------
    bool_hybrid = np.zeros_like(inside_pore, dtype=bool)
    bool_hybrid[inside_pore] = False
    bool_hybrid[inside_coating] = False
    bool_hybrid[hybrid_pore] = True
    bool_hybrid[hybrid_coating] = True

    grid.cell_data["bool_hybrid"] = bool_hybrid.flatten()

    # -------------------------------------------
    # Pore hybrid voxel labeling (hybrid or not)
    # -------------------------------------------
    bool_hybrid_pore = np.zeros_like(inside_pore, dtype=bool)
    bool_hybrid_pore[inside_pore] = False
    bool_hybrid_pore[inside_coating] = False
    bool_hybrid_pore[hybrid_pore] = True
    bool_hybrid_pore[hybrid_coating] = False

    grid.cell_data["bool_hybrid_pore"] = bool_hybrid_pore.flatten()

    # -------------------------------------------
    # Coating hybrid voxel labeling (hybrid or not)
    # -------------------------------------------
    bool_hybrid_coating = np.zeros_like(inside_pore, dtype=bool)
    bool_hybrid_coating[inside_pore] = False
    bool_hybrid_coating[inside_coating] = False
    bool_hybrid_coating[hybrid_pore] = False
    bool_hybrid_coating[hybrid_coating] = True

    grid.cell_data["bool_hybrid_coating"] = bool_hybrid_coating.flatten()

    # ------------------
    # Phases int and ext
    # ------------------
    phases_couple = np.zeros((*inside_pore.shape, 2), dtype=int)
    phases_couple[hybrid_pore] = (1, 2)
    phases_couple[hybrid_coating] = (2, 0)

    grid.cell_data["phases_couple"] = phases_couple.reshape(-1, 2)

    # ------------------------------------------------------------------------------------------------------------------
    # Volumic fraction labeling of the internal material,
    # if voxel at the edge between pore and coating, volumic fraction of the pore (the internal material of the voxel,
    # =1 for non-hybrid voxels
    #  + Local normal for each voxel in cartesian coordinates
    # -----------------------------------------------------------------------------------------------------------------
    volumic_fraction = np.ones_like(inside_pore, dtype=float)
    local_cartesian_normal = np.zeros(inside_pore.shape + (3,), dtype=float)

    indices = np.argwhere(bool_hybrid)
    origin = np.array(grid.origin)
    spacing = np.array(grid.spacing)

    for i, j, k in indices:
        # Local grid
        voxel_min = origin + spacing * np.array([i, j, k])
        voxel_max = voxel_min + spacing

        bounds_voxel = (
            voxel_min[0],
            voxel_max[0],
            voxel_min[1],
            voxel_max[1],
            voxel_min[2],
            voxel_max[2],
        )
        Nloc = 50  # Size of the local grid
        local_grid = make_grid(bounds_voxel, (Nloc, Nloc, Nloc))

        # Computation of the volumic fraction of the internal material in the voxel
        pts = local_grid.points
        r_local = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2 + pts[:, 2] ** 2)

        if bool_hybrid_pore[i, j, k] == True:
            fraction = np.sum(r_local < Rint) / pts.shape[0]

        elif bool_hybrid_coating[i, j, k] == True:  # hybrid_coating
            fraction = np.sum(r_local < Rint + coating_thickness) / pts.shape[0]

        voxel_center = voxel_min + 0.5 * spacing
        local_cartesian_normal[i, j, k] = voxel_center / np.linalg.norm(
            voxel_center
        )  # Computation hypothesis : spherical geometry centered on (0,0,0)

        volumic_fraction[i, j, k] = fraction

    # -----------------------------------------------------------------------------------------
    # Identification of 2 hybrid voxels (in order to test the laminate hybridation strategy)
    # -----------------------------------------------------------------------------------------
    bool_hybrid_test = np.zeros_like(inside_pore, dtype=bool)
    centers = grid.cell_centers().points  # (n_voxels, 3)

    # Target points
    targets = np.array([[-0.1, 0.05, 0.2], [0.1, 0.05, 0.2]])

    # For each point, find the closest voxel
    for target in targets:
        distances = np.linalg.norm(centers - target, axis=1)
        idx = np.argmin(distances)

        i, j, k = np.unravel_index(idx, inside_pore.shape)

        bool_hybrid_test[i, j, k] = True

    # Saving
    grid.cell_data["bool_hybrid_test"] = bool_hybrid_test.flatten()

    grid.cell_data["volumic_fraction"] = volumic_fraction.flatten()
    grid.cell_data["local_cartesian_normal"] = local_cartesian_normal.reshape((-1, 3))

    local_x = local_cartesian_normal.reshape((-1, 3))[:, 0]
    grid.cell_data["local_cartesian_normal_x"] = local_x

    local_y = local_cartesian_normal.reshape((-1, 3))[:, 1]
    grid.cell_data["local_cartesian_normal_y"] = local_y

    local_z = local_cartesian_normal.reshape((-1, 3))[:, 2]
    grid.cell_data["local_cartesian_normal_z"] = local_z

    print("Volume fractions:")
    print("matrix    :", np.mean(phase == 0))
    print("pore      :", np.mean(phase == 1))
    print("coating      :", np.mean(phase == 2))

    # -------------
    # Visualization
    # -------------

    # SLice at x = 0
    slice_mid = grid.slice(normal="x", origin=(0, 0, 0))

    # Plot phases
    plotter1 = pv.Plotter()
    plotter1.add_mesh(
        slice_mid,
        scalars="phase",
        show_edges=True,
        cmap=["lightblue", "purple", "blue"],
        clim=[0, 2],
    )
    pore_slice = pore.slice(normal="x", origin=(0, 0, 0))
    plotter1.add_mesh(
        pore_slice, color="black", opacity=1.0, show_edges=True, line_width=3
    )

    coating_slice = coating_outer.slice(normal="x", origin=(0, 0, 0))
    plotter1.add_mesh(
        coating_slice, color="black", opacity=1.0, show_edges=True, line_width=3
    )
    plotter1.add_title("Phases")
    plotter1.view_yz()
    plotter1.show()

    # Plot hybrid voxels
    plotter2 = pv.Plotter()
    plotter2.add_mesh(
        slice_mid, scalars="bool_hybrid", show_edges=True, cmap=["lightblue", "red"]
    )
    plotter2.add_mesh(
        pore_slice, color="black", opacity=1.0, show_edges=True, line_width=3
    )
    plotter2.add_mesh(
        coating_slice, color="black", opacity=1.0, show_edges=True, line_width=3
    )
    plotter2.add_title("Hybrid voxels")
    plotter2.view_yz()
    plotter2.show()

    # Plot volumic fractions and edges of the pore and the coating
    plotter3 = pv.Plotter()
    plotter3.add_mesh(
        slice_mid,
        scalars="volumic_fraction",
        show_edges=False,
        cmap="viridis",
        clim=[0, 1],
    )
    pore_slice = pore.slice(normal="x", origin=(0, 0, 0))
    plotter3.add_mesh(
        pore_slice, color="black", opacity=1.0, show_edges=True, line_width=3
    )

    coating_slice = coating_outer.slice(normal="x", origin=(0, 0, 0))
    plotter3.add_mesh(
        coating_slice, color="black", opacity=1.0, show_edges=True, line_width=3
    )
    plotter3.add_title("Volumic fraction")
    plotter3.view_yz()
    plotter3.show()

    grid.save(f"results/Hashin_num_{N}.vti")
    return grid, phase


def test_laminate_box():
    """
    2 boxes-geometry for test
    """

    N = 1
    tol = 1e-6
    bounds = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    bounds_hybrid = (-tol, tol, -0.5, 0.5, -0.5, 0.5)
    grid = make_grid(bounds, (N + 1, N + 1, N + 1))

    # Premier pavé (en bas sur X)
    box = pv.Box(bounds=bounds)
    box_hybrid = pv.Box(bounds=bounds_hybrid)
    inside_box = voxelize_on_grid(box, grid)
    inside_hybrid = voxelize_on_grid(box_hybrid, grid)

    # ------------------
    # Phase labeling
    # ------------------
    phase = np.zeros_like(inside_box, dtype=np.uint8)

    centers = grid.cell_centers().points

    phase = np.ones(centers.shape[0], dtype=np.uint8)

    # Condition : x < 0 -> 0, otherwise -> 1
    phase[centers[:, 0] >= 0] = 2

    grid.cell_data["phase"] = phase.flatten(order="F")

    # -------------------------------------------
    # Hybrid voxel labeling (hybrid or not)
    # -------------------------------------------
    bool_hybrid = np.zeros_like(inside_box, dtype=bool)
    bool_hybrid[inside_hybrid] = True

    grid.cell_data["bool_hybrid"] = bool_hybrid.flatten(order="F")

    # ------------------
    # Phases int and ext
    # ------------------
    phases_couple = np.zeros((*inside_box.shape, 2), dtype=int)
    phases_couple[inside_hybrid] = (1, 2)

    grid.cell_data["phases_couple"] = phases_couple.reshape(-1, 2, order="F")

    # ------------------------------------------------------------------------------------------------------------------
    # Volumic fraction labeling of the internal material,
    # if voxel at the edge between pore and coating, volumic fraction of the pore (the internal material of the voxel,
    # =1 for non-hybrid voxels
    #  + Local normal for each voxel in cartesian coordinates
    # -----------------------------------------------------------------------------------------------------------------
    volumic_fraction = np.ones_like(inside_box, dtype=float)
    local_cartesian_normal = np.zeros(inside_box.shape + (3,), dtype=float)

    indices = np.argwhere(bool_hybrid)

    for i, j, k in indices:
        local_cartesian_normal[i, j, k] = [1, 0, 0]
        volumic_fraction[i, j, k] = 0.5

    grid.cell_data["volumic_fraction"] = volumic_fraction.flatten(order="F")
    grid.cell_data["local_cartesian_normal"] = local_cartesian_normal.reshape(
        (-1, 3), order="F"
    )

    local_x = local_cartesian_normal.reshape((-1, 3), order="F")[:, 0]
    grid.cell_data["local_cartesian_normal_x"] = local_x

    local_y = local_cartesian_normal.reshape((-1, 3), order="F")[:, 1]
    grid.cell_data["local_cartesian_normal_y"] = local_y

    local_z = local_cartesian_normal.reshape((-1, 3), order="F")[:, 2]
    grid.cell_data["local_cartesian_normal_z"] = local_z

    grid.save(f"results/test_laminate_box_{N}.vti")
    return grid, phase


def main_multigrid_single_voxel():
    """
    Voxelized box with all phases = 0 except center voxel with phase = 1
    """

    N = 3
    bounds = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)
    grid = make_grid(bounds, (N + 1, N + 1, N + 1))

    box = pv.Box(bounds=bounds)
    inside_box = voxelize_on_grid(box, grid)

    # ------------------
    # Phase labeling
    # ------------------
    phase = np.zeros_like(inside_box, dtype=np.uint8)

    centers = grid.cell_centers().points

    dist = np.linalg.norm(centers, axis=1)
    idx_center = np.argmin(dist)

    phase = np.zeros(centers.shape[0], dtype=np.uint8)
    phase[idx_center] = 1

    grid.cell_data["phase"] = phase.flatten(order="F")

    grid.save(f"results/multigrid_{N}.vti")
    return grid, phase


def test_laminate_box():

    N = 1  # Number of voxels in each direction

    bounds = (-0.5, 0.5, -0.5, 0.5, -0.5, 0.5)  # Boundaries of the domain
    grid = make_grid(bounds, (N + 1, N + 1, N + 1))

    box = pv.Box(bounds=bounds)
    inside_box = voxelize_on_grid(box, grid)

    # ------------------
    # Phase labeling
    # ------------------
    # 0 = matrix, 1 = pore, 2 = coating
    phase = np.zeros_like(inside_box, dtype=np.uint8)

    grid.cell_data["phase"] = phase.flatten()

    # -------------------------------------------
    # Hybrid voxel labeling (hybrid or not)
    # -------------------------------------------
    bool_hybrid = np.zeros_like(inside_box, dtype=bool)
    bool_hybrid[inside_box] = True

    grid.cell_data["bool_hybrid"] = bool_hybrid.flatten()

    # ------------------
    # Phases int and ext
    # ------------------
    phases_couple = np.zeros((*inside_box.shape, 2), dtype=int)
    phases_couple[inside_box] = (1, 2)

    grid.cell_data["phases_couple"] = phases_couple.reshape(-1, 2)

    # ------------------------------------------------------------------------------------------------------------------
    # Volumic fraction labeling of the internal material,
    # if voxel at the edge between pore and coating, volumic fraction of the pore (the internal material of the voxel,
    # =1 for non-hybrid voxels
    #  + Local normal for each voxel in cartesian coordinates
    # -----------------------------------------------------------------------------------------------------------------
    volumic_fraction = np.ones_like(inside_box, dtype=float)
    local_cartesian_normal = np.zeros(inside_box.shape + (3,), dtype=float)

    indices = np.argwhere(bool_hybrid)
    origin = np.array(grid.origin)
    spacing = np.array(grid.spacing)

    for i, j, k in indices:
        local_cartesian_normal[i, j, k] = [
            1,
            0,
            0,
        ]  # Computation hypothesis : spherical geometry centered on (0,0,0)
        volumic_fraction[i, j, k] = 0.3

    grid.cell_data["volumic_fraction"] = volumic_fraction.flatten()
    grid.cell_data["local_cartesian_normal"] = local_cartesian_normal.reshape((-1, 3))

    local_x = local_cartesian_normal.reshape((-1, 3))[:, 0]
    grid.cell_data["local_cartesian_normal_x"] = local_x

    local_y = local_cartesian_normal.reshape((-1, 3))[:, 1]
    grid.cell_data["local_cartesian_normal_y"] = local_y

    local_z = local_cartesian_normal.reshape((-1, 3))[:, 2]
    grid.cell_data["local_cartesian_normal_z"] = local_z

    grid.save(f"results/test_laminate_{N}.vti")
    return grid, phase


# main_3_phases()
# main_Hashin_an()
main_Hashin_num()
# main_multigrid_single_voxel()
# test_laminate_box()
