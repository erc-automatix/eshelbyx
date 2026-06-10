# Importing FEniCSx libraries
from pathlib import Path
import os, ufl, math
import numpy as np
import pandas as pd
import dolfinx_mpc.utils
from dolfinx import fem, io
from dolfinx_mpc import LinearProblem
from mesh_cube_sphere import create_mesh_cube_sphere
from macroscopic_strain import define_macroscopic_strain
from material_properties import define_material_properties


# Current folder
curr_dir = Path(__file__).resolve().parent

# Defining parameters and dimensions
L = 1.0  # Edge of the cube
gdim = 3  # Geometric dimension of the domain
fdim = 2  # Facet dimension
f_values = np.linspace(0.05, 0.52, num=20)  # Different values ​​of volume fraction


for f in f_values:

    R = ((3 * f * L**3) / (4 * math.pi)) ** (1 / 3)

    print(f"\n--- Treatment for f = {f:.3f} → R = {R:.3f} ---")

    out_fname = curr_dir / "results" / f"Cube_Sphere_Exact_PBC_f{f:.3f}"
    if not os.path.exists(out_fname):
        os.makedirs(out_fname)

    print(f"\n Creating a mesh for a sphere with radius R={R}")

    # Creation of the mesh
    domain, cells, facets = create_mesh_cube_sphere(R, L, gdim, fdim)

    # Saving the mesh in XDMF format (for viewing in ParaView)
    with io.XDMFFile(domain.comm, out_fname / "mesh.xdmf", "w") as xdmf:
        xdmf.write_mesh(domain)

        print(f"\n 3D mesh for R={R:.3f} generated and saved in 'mesh.xdmf'")

        # Calculation of volumes and volume fraction of inclusion
        dx = ufl.Measure("dx", subdomain_data=cells, domain=domain)
        vol = fem.assemble_scalar(fem.form(1 * dx))
        vol_inc = fem.assemble_scalar(fem.form(1 * dx(2)))
        vol_mat = fem.assemble_scalar(fem.form(1 * dx(1)))
        frac_vol = vol_inc / vol

        print(f"Radius of the sphere for f={f} : {R}")
        print(f"Total volume for f={f} : {vol}")
        print(f"Volume inclusion for f={f} : {vol_inc}")
        print(f"Matrix volume for f={f} : {vol_mat}")
        print(f"Inclusion volume fraction for f={f} : {frac_vol}")

        # Variational formulation with heterogeneous material properties

        # Definition of mechanical properties
        E, nu, mu, K, lmbda = define_material_properties(domain, cells)

        # Definition of macroscopic deformation
        Eps, Eps_, y = define_macroscopic_strain(domain)

        # Variational formulation

        def epsilon(v):
            return ufl.sym(ufl.grad(v))

        def sigma(v):
            eps = Eps + epsilon(v)
            return lmbda * ufl.tr(eps) * ufl.Identity(gdim) + 2 * mu * eps

        V = fem.functionspace(domain, ("P", 2, (gdim,)))
        du = ufl.TrialFunction(V)
        u_ = ufl.TestFunction(V)
        a_form, L_form = ufl.system(ufl.inner(sigma(du), epsilon(u_)) * ufl.dx)

        # Periodic boundary conditions enforcement using `dolfinx_mpc`

        # We must now define the periodic boundary conditions for the fluctuation field. For that, we make use of `dolfinx_mpc` providing a `MultiPointConstraint` object which can account for periodicity conditions. Note that periodic conditions do not fix rigid body translations. To remove them we choose here, for simplicity, to fix the displacement of the single point of coordinate `(0, 0)`. An alternative can consist of introducing constant Lagrange multipliers as discussed in the legacy demo. This solution requires however to use `Real` elements which require special care and are currently available in the `scifem` package https://github.com/scientificcomputing/scifem.

        # Definition of faces

        def right(x):
            return np.isclose(x[0], L) & np.logical_not(
                np.isclose(x[1], 0)
                | np.isclose(x[2], 0)
                | np.isclose(x[1], L)
                | np.isclose(x[2], L)
            )

        def left(x):
            return np.isclose(x[0], 0) & np.logical_not(
                np.isclose(x[1], 0)
                | np.isclose(x[2], 0)
                | np.isclose(x[1], L)
                | np.isclose(x[2], L)
            )

        def top(x):
            return np.isclose(x[1], L) & np.logical_not(
                np.isclose(x[0], 0)
                | np.isclose(x[2], 0)
                | np.isclose(x[0], L)
                | np.isclose(x[2], L)
            )

        def bottom(x):
            return np.isclose(x[1], 0) & np.logical_not(
                np.isclose(x[0], 0)
                | np.isclose(x[2], 0)
                | np.isclose(x[0], L)
                | np.isclose(x[2], L)
            )

        def front(x):
            return np.isclose(x[2], L) & np.logical_not(
                np.isclose(x[0], 0)
                | np.isclose(x[1], 0)
                | np.isclose(x[0], L)
                | np.isclose(x[1], L)
            )

        def back(x):
            return np.isclose(x[2], 0) & np.logical_not(
                np.isclose(x[0], 0)
                | np.isclose(x[1], 0)
                | np.isclose(x[0], L)
                | np.isclose(x[1], L)
            )

        # Definition of the edges of the upper face

        def left_top_edge(x):
            return (
                np.isclose(x[0], 0)
                & np.isclose(x[1], L)
                & np.logical_not(np.isclose(x[2], 0) | np.isclose(x[2], L))
            )

        def top_back_edge(x):
            return (
                np.isclose(x[1], L)
                & np.isclose(x[2], 0)
                & np.logical_not(np.isclose(x[0], 0) | np.isclose(x[0], L))
            )

        def right_top_edge(x):
            return (
                np.isclose(x[0], L)
                & np.isclose(x[1], L)
                & np.logical_not(np.isclose(x[2], 0) | np.isclose(x[2], L))
            )

        def top_front_edge(x):
            return (
                np.isclose(x[1], L)
                & np.isclose(x[2], L)
                & np.logical_not(np.isclose(x[0], 0) | np.isclose(x[0], L))
            )

        # Definition of the edges of the lower face

        def left_bottom_edge(x):
            return (
                np.isclose(x[0], 0)
                & np.isclose(x[1], 0)
                & np.logical_not(np.isclose(x[2], 0) | np.isclose(x[2], L))
            )

        def bottom_back_edge(x):
            return (
                np.isclose(x[1], 0)
                & np.isclose(x[2], 0)
                & np.logical_not(np.isclose(x[0], 0) | np.isclose(x[0], L))
            )

        def right_bottom_edge(x):
            return (
                np.isclose(x[0], L)
                & np.isclose(x[1], 0)
                & np.logical_not(np.isclose(x[2], 0) | np.isclose(x[2], L))
            )

        def bottom_front_edge(x):
            return (
                np.isclose(x[1], 0)
                & np.isclose(x[2], L)
                & np.logical_not(np.isclose(x[0], 0) | np.isclose(x[0], L))
            )

        # Definition of the edges of the front face

        def left_front_edge(x):
            return (
                np.isclose(x[0], 0)
                & np.isclose(x[2], L)
                & np.logical_not(np.isclose(x[1], 0) | np.isclose(x[1], L))
            )

        def right_front_edge(x):
            return (
                np.isclose(x[0], L)
                & np.isclose(x[2], L)
                & np.logical_not(np.isclose(x[1], 0) | np.isclose(x[1], L))
            )

        # Definition of the edges of the back face

        def left_back_edge(x):
            return (
                np.isclose(x[0], 0)
                & np.isclose(x[2], 0)
                & np.logical_not(np.isclose(x[1], 0) | np.isclose(x[1], L))
            )

        def right_back_edge(x):
            return (
                np.isclose(x[0], L)
                & np.isclose(x[2], 0)
                & np.logical_not(np.isclose(x[1], 0) | np.isclose(x[1], L))
            )

        # Periodic relation between faces

        def periodic_relation_left_right(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0] - L
            out_x[1] = x[1]
            out_x[2] = x[2]
            return out_x

        def periodic_relation_bottom_top(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0]
            out_x[1] = x[1] - L
            out_x[2] = x[2]
            return out_x

        def periodic_relation_front_back(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0]
            out_x[1] = x[1]
            out_x[2] = x[2] - L
            return out_x

        # Periodic relation of the edges of the upper face

        def periodic_relation_left_top_edge(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0]
            out_x[1] = x[1] - L
            out_x[2] = x[2]
            return out_x

        def periodic_relation_top_back_edge(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0]
            out_x[1] = x[1] - L
            out_x[2] = x[2]
            return out_x

        def periodic_relation_right_top_edge(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0] - L
            out_x[1] = x[1] - L
            out_x[2] = x[2]
            return out_x

        def periodic_relation_top_front_edge(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0]
            out_x[1] = x[1] - L
            out_x[2] = x[2] - L
            return out_x

        # Periodic relation of the edges of the lower face

        def periodic_relation_right_bottom_edge(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0] - L
            out_x[1] = x[1]
            out_x[2] = x[2]
            return out_x

        def periodic_relation_bottom_front_edge(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0]
            out_x[1] = x[1]
            out_x[2] = x[2] - L
            return out_x

        # Periodic relation of the edges of the front face

        def periodic_relation_left_front_edge(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0]
            out_x[1] = x[1]
            out_x[2] = x[2] - L
            return out_x

        def periodic_relation_right_front_edge(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0] - L
            out_x[1] = x[1]
            out_x[2] = x[2] - L
            return out_x

        # Periodic relation of the edges of the back face

        def periodic_relation_right_back_edge(x):
            out_x = np.zeros(x.shape)
            out_x[0] = x[0] - L
            out_x[1] = x[1]
            out_x[2] = x[2]
            return out_x

        # Boundary conditions

        point_dof1 = fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0) & np.isclose(x[2], 0)
        )
        point_dof2 = fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], L) & np.isclose(x[1], 0) & np.isclose(x[2], 0)
        )
        point_dof3 = fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], 0) & np.isclose(x[1], L) & np.isclose(x[2], 0)
        )
        point_dof4 = fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], 0) & np.isclose(x[1], 0) & np.isclose(x[2], L)
        )
        point_dof5 = fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], 0) & np.isclose(x[1], L) & np.isclose(x[2], L)
        )
        point_dof6 = fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], L) & np.isclose(x[1], 0) & np.isclose(x[2], L)
        )
        point_dof7 = fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], L) & np.isclose(x[1], L) & np.isclose(x[2], 0)
        )
        point_dof8 = fem.locate_dofs_geometrical(
            V, lambda x: np.isclose(x[0], L) & np.isclose(x[1], L) & np.isclose(x[2], L)
        )

        bc1 = fem.dirichletbc(np.zeros((gdim,)), point_dof1, V)
        bc2 = fem.dirichletbc(np.zeros((gdim,)), point_dof2, V)
        bc3 = fem.dirichletbc(np.zeros((gdim,)), point_dof3, V)
        bc4 = fem.dirichletbc(np.zeros((gdim,)), point_dof4, V)
        bc5 = fem.dirichletbc(np.zeros((gdim,)), point_dof5, V)
        bc6 = fem.dirichletbc(np.zeros((gdim,)), point_dof6, V)
        bc7 = fem.dirichletbc(np.zeros((gdim,)), point_dof7, V)
        bc8 = fem.dirichletbc(np.zeros((gdim,)), point_dof8, V)

        bcs = [bc1, bc2, bc3, bc4, bc5, bc6, bc7, bc8]

        mpc = dolfinx_mpc.MultiPointConstraint(V)

        # MPC between faces
        mpc.create_periodic_constraint_geometrical(
            V, right, periodic_relation_left_right, bcs
        )
        mpc.create_periodic_constraint_geometrical(
            V, top, periodic_relation_bottom_top, bcs
        )
        mpc.create_periodic_constraint_geometrical(
            V, front, periodic_relation_front_back, bcs
        )

        # MPC edges upper face
        mpc.create_periodic_constraint_geometrical(
            V, left_top_edge, periodic_relation_left_top_edge, bcs
        )

        mpc.create_periodic_constraint_geometrical(
            V, top_back_edge, periodic_relation_top_back_edge, bcs
        )

        mpc.create_periodic_constraint_geometrical(
            V, right_top_edge, periodic_relation_right_top_edge, bcs
        )

        mpc.create_periodic_constraint_geometrical(
            V, top_front_edge, periodic_relation_top_front_edge, bcs
        )

        # MPC edges lower face
        mpc.create_periodic_constraint_geometrical(
            V, right_bottom_edge, periodic_relation_right_bottom_edge, bcs
        )

        mpc.create_periodic_constraint_geometrical(
            V, bottom_front_edge, periodic_relation_bottom_front_edge, bcs
        )

        # MPC edges front face
        mpc.create_periodic_constraint_geometrical(
            V, left_front_edge, periodic_relation_left_front_edge, bcs
        )

        mpc.create_periodic_constraint_geometrical(
            V, right_front_edge, periodic_relation_right_front_edge, bcs
        )

        # MPC edges back face
        mpc.create_periodic_constraint_geometrical(
            V, right_back_edge, periodic_relation_right_back_edge, bcs
        )

        mpc.finalize()

        # Problem resolution:
        u = fem.Function(mpc.function_space, name="Displacement")
        v = fem.Function(mpc.function_space, name="Periodic_fluctuation")

        petsc_options = petsc_options = {"ksp_type": "cg", "pc_type": "gamg"}

        problem = LinearProblem(
            a_form, L_form, mpc, bcs=bcs, u=v, petsc_options=petsc_options
        )

        elementary_load = [
            np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]]),
            np.array(
                [
                    [0.0, math.sqrt(2) / 2, 0.0],
                    [math.sqrt(2) / 2, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ]
            ),
            np.array(
                [
                    [0.0, 0.0, math.sqrt(2) / 2],
                    [0.0, 0.0, 0.0],
                    [math.sqrt(2) / 2, 0.0, 0.0],
                ]
            ),
            np.array(
                [
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, math.sqrt(2) / 2],
                    [0.0, math.sqrt(2) / 2, 0.0],
                ]
            ),
        ]
        load_labels = ["Exx", "Eyy", "Ezz", "Exy", "Exz", "Eyz"]
        dim_load = len(elementary_load)

        C_hom = np.zeros((dim_load, dim_load))

        disp_file = io.VTKFile(domain.comm, out_fname / "Displacement.pvd", "w")
        stress_file = io.VTKFile(domain.comm, out_fname / "Stress.pvd", "w")
        strain_file = io.VTKFile(domain.comm, out_fname / "Strain.pvd", "w")

        for nload in range(dim_load):
            Eps.value = elementary_load[nload]
            u.interpolate(
                fem.Expression(ufl.dot(Eps, y), V.element.interpolation_points())
            )
            problem.solve()
            u.x.array[:] += v.x.array[:]

            # Recording of movement fields
            disp_file.write_function(u, nload)

            for nload_ in range(dim_load):
                Eps_.value = elementary_load[nload_]
                C_hom[nload, nload_] = (
                    fem.assemble_scalar(fem.form(ufl.inner(sigma(v), Eps_) * ufl.dx))
                    / vol
                )

                # Calculation and recording of stress and strain fields
                V0 = fem.functionspace(domain, ("DG", 0, (gdim, gdim)))
                stress_exp = fem.Expression(sigma(v), V0.element.interpolation_points())
                strain_exp = fem.Expression(
                    Eps + epsilon(v), V0.element.interpolation_points()
                )
                stress = fem.Function(V0, name="Stress_3D")
                strain = fem.Function(V0, name="Strain_3D")
                stress.interpolate(stress_exp)
                strain.interpolate(strain_exp)

                stress_file.write_function(stress, nload)
                strain_file.write_function(strain, nload)

            # Closing files
            disp_file.close()
            stress_file.close()
            strain_file.close()

        # Displaying the homogenization matrix
        with np.printoptions(suppress=True, precision=0):
            print(f"\n C_hom for f={f}:\n")
            print(C_hom)

        # Reading in table form
        df_C_hom = pd.DataFrame(C_hom)
        print(f"\n Display C_hom as a table for f = {f:.3f}")
        print("\n C_hom =")
        print(df_C_hom.to_string(formatters={"float": "{:.6}".format}))

        # Save in Excel format
        df_C_hom.to_excel(out_fname / f"C_hom_f{f:.3f}.xlsx", index=False, header=False)

        lmbda_hom = C_hom[0, 1]
        mu_hom = C_hom[3, 3] / 2
        E_hom = mu_hom * (3 * lmbda_hom + 2 * mu_hom) / (lmbda_hom + mu_hom)
        nu_hom = lmbda_hom / (2 * (lmbda_hom + mu_hom))

        print(
            f"\n Isotropy check for f = {f}: C11 = {C_hom[0, 0] : .0f}, lmbda_hom+2mu_hom = {lmbda_hom + 2 * mu_hom : .0f}"
        )
        print(f"\n Apparent Young modulus for f = {f}: {E_hom:.0f}")
        print(f"\n Apparent Poisson ratio for f = {f}: {nu_hom:.3f}")
