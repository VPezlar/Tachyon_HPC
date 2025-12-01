import gmsh
import sys


def gmshMesh(x1, x2, x3, x4, y1, y2, y3, y4):
    ##################################### GMSH Section
    # Initialize gmsh:
    gmsh.initialize()

    # points:
    lc = 1

    # occ stands for GMSH OpenCascade module (Works better for this case)
    # (One for the whole project i.e. cannot use occ for points and standard for rest)
    P1 = gmsh.model.occ.add_point(x2[0], y2[0], 0, lc)
    P2 = gmsh.model.occ.add_point(x2[0], y2[-1], 0, lc)
    P3 = gmsh.model.occ.add_point(x2[-1], y2[-1], 0, lc)
    P4 = gmsh.model.occ.add_point(x3[int((len(x3)-1)/2)], y3[-1], 0, lc)
    P5 = gmsh.model.occ.add_point(x3[-1], y3[-1], 0, lc)
    P6 = gmsh.model.occ.add_point(x4[-1], y4[-1], 0, lc)
    P7 = gmsh.model.occ.add_point(x4[-1], y4[0], 0, lc)
    P8 = gmsh.model.occ.add_point(x4[0], y4[0], 0, lc)
    P9 = gmsh.model.occ.add_point(x1[-1], y1[0], 0, lc)
    P10 = gmsh.model.occ.add_point(x1[int((len(x1)-1)/2)], y1[0], 0, lc)
    P11 = gmsh.model.occ.add_point(x1[0], y1[0], 0, lc)
    P12 = gmsh.model.occ.add_point(x1[0], y1[-1], 0, lc)
    P13 = gmsh.model.occ.add_point(x1[int((len(x1)-1)/2)], y1[-1], 0, lc)

    # Defining Lines from said points
    L1 = gmsh.model.occ.add_line(P1, P2)
    L2 = gmsh.model.occ.add_line(P2, P3)
    L3 = gmsh.model.occ.add_line(P3, P4)
    L4 = gmsh.model.occ.add_line(P4, P5)
    L5 = gmsh.model.occ.add_line(P5, P6)
    L6 = gmsh.model.occ.add_line(P6, P7)
    L7 = gmsh.model.occ.add_line(P7, P8)
    L8 = gmsh.model.occ.add_line(P8, P9)
    L9 = gmsh.model.occ.add_line(P9, P10)
    L10 = gmsh.model.occ.add_line(P10, P11)
    L11 = gmsh.model.occ.add_line(P11, P12)
    L12 = gmsh.model.occ.add_line(P12, P1)
    L13 = gmsh.model.occ.add_line(P12, P3)
    L14 = gmsh.model.occ.add_line(P13, P4)
    L15 = gmsh.model.occ.add_line(P8, P5)
    L16 = gmsh.model.occ.add_line(P8, P13)
    L17 = gmsh.model.occ.add_line(P13, P10)
    L18 = gmsh.model.occ.add_line(P13, P12)

    # Defining curve loops for later definition of surfaces
    loop1 = gmsh.model.occ.add_curve_loop([L1, L2, L13, L12])
    loop2 = gmsh.model.occ.add_curve_loop([L13, L3, L14, L18])
    loop3 = gmsh.model.occ.add_curve_loop([L14, L4, L15, L16])
    loop4 = gmsh.model.occ.add_curve_loop([L15, L5, L6, L7])
    loop5 = gmsh.model.occ.add_curve_loop([L17, L16, L8, L9])
    loop6 = gmsh.model.occ.add_curve_loop([L11, L18, L17, L10])

    # Defining Surfaces
    Face1 = gmsh.model.occ.add_plane_surface([loop1])
    Face2 = gmsh.model.occ.add_plane_surface([loop2])
    Face3 = gmsh.model.occ.add_plane_surface([loop3])
    Face4 = gmsh.model.occ.add_plane_surface([loop4])
    Face5 = gmsh.model.occ.add_plane_surface([loop5])
    Face6 = gmsh.model.occ.add_plane_surface([loop6])

    # Command synchronise() creates structures from the definitions above.
    # Necessary for following operations (i.e. objects must exist before operations are performed on them)
    gmsh.model.occ.synchronize()

    TF1 = gmsh.model.mesh.set_transfinite_surface(Face1, "Left", [P1, P2, P3, P12])
    TF2 = gmsh.model.mesh.set_transfinite_surface(Face2, "Left", [P12, P3, P4, P13])
    TF3 = gmsh.model.mesh.set_transfinite_surface(Face3, "Left", [P13, P4, P5, P8])
    TF4 = gmsh.model.mesh.set_transfinite_surface(Face4, "Left", [P8, P5, P6, P7])
    TF5 = gmsh.model.mesh.set_transfinite_surface(Face5, "Left", [P10, P13, P8, P9])
    TF6 = gmsh.model.mesh.set_transfinite_surface(Face6, "Left", [P11, P12, P13, P10])

    gmsh.model.occ.synchronize()

    res = 51  # How many points (nodes) are on specified line
    # First a line of choice is specified, then res.. and then distribution of said points along the line
    gmsh.model.mesh.setTransfiniteCurve(L1, res * 5, "Progression", -0.98)
    gmsh.model.mesh.setTransfiniteCurve(L2, res, "Progression", 1)
    gmsh.model.mesh.setTransfiniteCurve(L3, res, "Progression", -0.9)
    gmsh.model.mesh.setTransfiniteCurve(L4, res, "Progression", 0.9)
    gmsh.model.mesh.setTransfiniteCurve(L5, res * 5, "Progression", 1)
    gmsh.model.mesh.setTransfiniteCurve(L6, res * 5, "Progression", 0.98)
    gmsh.model.mesh.setTransfiniteCurve(L7, res * 5, "Progression", 1)
    gmsh.model.mesh.setTransfiniteCurve(L8, res, "Progression", 0.9)
    gmsh.model.mesh.setTransfiniteCurve(L9, res, "Progression", -0.9)
    gmsh.model.mesh.setTransfiniteCurve(L10, res, "Progression", 0.9)
    gmsh.model.mesh.setTransfiniteCurve(L11, res, "Progression", -0.9)
    gmsh.model.mesh.setTransfiniteCurve(L12, res, "Progression", 1)
    gmsh.model.mesh.setTransfiniteCurve(L13, res * 5, "Progression", -0.98)
    gmsh.model.mesh.setTransfiniteCurve(L14, res * 5, "Progression", -0.98)
    gmsh.model.mesh.setTransfiniteCurve(L15, res * 5, "Progression", -0.98)
    gmsh.model.mesh.setTransfiniteCurve(L16, res, "Progression", -0.9)
    gmsh.model.mesh.setTransfiniteCurve(L17, res, "Progression", 0.9)
    gmsh.model.mesh.setTransfiniteCurve(L18, res, "Progression", 0.9)

    gmsh.option.setNumber("Mesh.RecombineAll", 2)

    # Command synchronise() creates structures from the definitions above.
    gmsh.model.occ.synchronize()

    # Creating Physical Entities for BC definition in CFD software of choice (Assigning tags)
    # Surfaces
    Air = gmsh.model.addPhysicalGroup(2, [Face1, Face2, Face3, Face4, Face5, Face6], name="Air")

    # Edges
    # Inlet
    inlet = gmsh.model.addPhysicalGroup(1, [L1], name="Inlet")
    # Outlet
    outlet = gmsh.model.addPhysicalGroup(1, [L2, L3, L4, L5, L6], name="Outlet")
    # Wall
    wall = gmsh.model.addPhysicalGroup(1, [L12, L11, L10, L9, L8, L7], name="Wall")

    # # Command synchronise() creates structures from the definitions above.
    gmsh.model.occ.synchronize()

    # Create mesh (2 for 2D, 3 for 3D...)
    gmsh.model.mesh.generate(2)

    # Generate mesh data and use ASCII2 format (for OpenFOAM)
    # gmsh.option.setNumber("Mesh.MshFileVersion", 2)

    # Setting to save all entities
    gmsh.option.setNumber("Mesh.SaveAll", 0)
    gmsh.write("/Users/vojtechpezlar/Documents/Projects/Multi-domain Cavity/SU2/Square/Cavity_Square.su2")

    # Creates  graphical user interface
    if 'close' not in sys.argv:
        gmsh.fltk.run()

    # It finalizes the Gmsh API
    gmsh.finalize()