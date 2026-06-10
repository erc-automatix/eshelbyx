# Importing FEniCSx libraries
import gmsh
import numpy as np
from mpi4py import MPI
from dolfinx.io.gmshio import model_to_mesh

# Definition of dimensions


def create_mesh_cube_sphere(R, L, gdim, fdim):

    gmsh.initialize()

    gmsh.option.setNumber("General.Terminal", 1)

    gmsh.model.add(f"cube_with_sphere_R{R}")

    occ = gmsh.model.occ

    # Definition of cube
    cube = occ.addBox(0, 0, 0, L, L, L)

    # Definition of the sphere
    sphere = occ.addSphere(L / 2, L / 2, L / 2, R)

    # Subtraction: Cube - Sphere
    cut, _ = occ.cut(
        [(gdim, cube)], [(gdim, sphere)], removeObject=True, removeTool=False
    )
    occ.synchronize()

    # Defining fields for mesh size
    h_sphere = R / 4
    h_cube = L / 8

    field_tag = gmsh.model.mesh.field.add("Ball")
    gmsh.model.mesh.field.setNumber(field_tag, "VIn", h_sphere)
    gmsh.model.mesh.field.setNumber(field_tag, "VOut", h_cube)
    gmsh.model.mesh.field.setNumber(field_tag, "XCenter", L / 2)
    gmsh.model.mesh.field.setNumber(field_tag, "YCenter", L / 2)
    gmsh.model.mesh.field.setNumber(field_tag, "ZCenter", L / 2)
    gmsh.model.mesh.field.setNumber(field_tag, "Radius", R)
    gmsh.model.mesh.field.setNumber(field_tag, "Thickness", (L - 2 * R) / 4)
    gmsh.model.mesh.field.setAsBackgroundMesh(field_tag)

    occ.synchronize()

    # Defining volumes as physical groupsDefining fields for mesh size
    gmsh.model.addPhysicalGroup(gdim, [cut[0][1]], 1)  # Matrice
    gmsh.model.setPhysicalName(gdim, 1, "Matrix")
    gmsh.model.addPhysicalGroup(gdim, [sphere], 2)  # Inclusion
    gmsh.model.setPhysicalName(gdim, 2, "Inclusion")

    # Identification of surfaces
    surfaces = gmsh.model.getEntities(fdim)
    print(surfaces)

    # Consistent marking of surfaces
    for tag, name, id in zip(
        [10, 12, 11, 9, 8, 13, 7],
        ["front", "back", "top", "bottom", "left", "right", "interface"],
        [1, 2, 3, 4, 5, 6, 7],
    ):
        gmsh.model.addPhysicalGroup(fdim, [tag], id)
        gmsh.model.setPhysicalName(fdim, id, name)

    # Definition of periodic translations
    translation_x = np.array([[1, 0, 0, L], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    translation_y = np.array([[1, 0, 0, 0], [0, 1, 0, L], [0, 0, 1, 0], [0, 0, 0, 1]])
    translation_z = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, L], [0, 0, 0, 1]])

    try:
        gmsh.model.mesh.set_periodic(fdim, [13], [8], translation_x.flatten())
        gmsh.model.mesh.set_periodic(fdim, [11], [9], translation_y.flatten())
        gmsh.model.mesh.set_periodic(fdim, [10], [12], translation_z.flatten())
    except Exception as e:
        print(f"Error setting periodic conditions: {e}")

    # Mesh generation
    gmsh.model.mesh.generate(gdim)

    # Displaying mesh entities
    print(f"3-dimensional entities for R={R:.3f}:", gmsh.model.getEntities(gdim))
    print(f"2-dimensional entities for R={R:.3f}:", gmsh.model.getEntities(fdim))

    # Save mesh and geometry
    # gmsh.write(f"cube_with_sphere_R{R:.3f}.msh")
    # gmsh.write(f"cube_with_sphere_R{R:.3f}.geo_unrolled")

    # Conversion for dolfinx
    domain, cells, facets = model_to_mesh(gmsh.model, MPI.COMM_WORLD, 0, gdim=gdim)

    # Displaying the GMSH interface
    # gmsh.fltk.run()

    # GMSH cleaning
    gmsh.finalize()

    print(f"Mesh generated for R={R} : cube_with_sphere_R{R:.3f}.msh")

    return domain, cells, facets
