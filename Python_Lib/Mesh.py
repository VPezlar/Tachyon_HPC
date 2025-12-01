import gmsh
import sys

import numpy as np


def gmshMesh(x1, x2, x3, x4, x5, y1, y2, y3, y4, y5, Intersect):
    ##################################### GMSH Section
    # Initialize gmsh:
    gmsh.initialize()
    # points:
    lc = 1

    # Domain 1 definition of points (Lower Spline)
    Spline1 = []
    for i in range(len(x1)-2):
        Spline1.append(gmsh.model.geo.addPoint(x1[0, i+1], y1[0, i+1], 0, lc))
        # Spline1.append(gmsh.model.occ.add_point(x1[i], y1[0, i], 0, lc))

    # Domain 1 definition of points (Upper Spline)
    Spline2 = []
    for i in range(len(x1)-2):
        Spline2.append(gmsh.model.geo.addPoint(x1[-1, i+1], y1[-1, i+1], 0, lc))

    Spline1.insert(0, gmsh.model.geo.add_point(x1[0, 0], y1[0, 0], 0, lc))
    Spline1.append(gmsh.model.geo.add_point(x1[0, -1], y1[0, -1], 0, lc))

    Spline2.append(gmsh.model.geo.add_point(x1[-1, -1], y1[-1, -1], 0, lc))
    Spline2.insert(0, gmsh.model.geo.add_point(x1[-1, 0], y1[-1, 0], 0, lc))
    # P1 = gmsh.model.geo.add_point(x1[0, 0], y1[0, 0], 0, lc)
    # P2 = gmsh.model.geo.add_point(x1[0, -1], y1[0, -1], 0, lc)
    # P3 = gmsh.model.geo.add_point(x1[-1, -1], y1[-1, -1], 0, lc)
    # P4 = gmsh.model.geo.add_point(x1[-1, 0], y1[-1, 0], 0, lc)
    # P9 = gmsh.model.geo.add_point(x3[-1], y3[-1], 0, lc)
    # P10 = gmsh.model.geo.add_point(x4[-1, -1], y4[-1, -1], 0, lc)
    # P11 = gmsh.model.geo.add_point(x5[-1], y5[-1], 0, lc)

    L1 = gmsh.model.geo.add_spline(Spline1)
    L3 = gmsh.model.geo.add_spline(Spline2)

    # # Defining curve loops for later definition of surfaces
    # loop1 = gmsh.model.occ.add_curve_loop([L1, L2, L9, L8])
    # loop2 = gmsh.model.occ.add_curve_loop([L9, L3, L4, L10])
    # loop3 = gmsh.model.occ.add_curve_loop([L7, L10, L5, L6])
    # #
    # # # Defining Surfaces
    # Face1 = gmsh.model.occ.add_plane_surface([loop1])
    # Face2 = gmsh.model.occ.add_plane_surface([loop2])
    # Face3 = gmsh.model.occ.add_plane_surface([loop3])

    # # Command synchronise() creates structures from the definitions above.
    # # Necessary for following operations (i.e. objects must exist before operations are performed on them)
    # gmsh.model.geo.synchronize()
    #
    # TF1 = gmsh.model.mesh.set_transfinite_surface(Face1, "Left", [P1, P2, P3, Spline2[0]])
    # TF2 = gmsh.model.mesh.set_transfinite_surface(Face2, "Left", [Spline2[0], P3, P4, Spline2[-1]])
    # TF3 = gmsh.model.mesh.set_transfinite_surface(Face3, "Left", [Spline1[0], Spline2[0], Spline2[-1], Spline1[-1]])

    gmsh.model.geo.synchronize()

    res = 21
    # gmsh.model.mesh.setTransfiniteCurve(L1, res + 100, "Progression", 1)
    # gmsh.model.mesh.setTransfiniteCurve(L2, res, "Progression", 1)
    # gmsh.model.mesh.setTransfiniteCurve(L3, res, "Progression", 1)
    # gmsh.model.mesh.setTransfiniteCurve(L4, res, "Progression", 1)
    #
    # gmsh.model.mesh.setTransfiniteCurve(L5, res, "Progression", 1)
    # gmsh.model.mesh.setTransfiniteCurve(L6, res + 200, "Progression", 1)
    # gmsh.model.mesh.setTransfiniteCurve(L7, res + 21, "Progression", 1)
    # #
    # gmsh.model.mesh.setTransfiniteCurve(L8, res, "Progression", 1)
    # gmsh.model.mesh.setTransfiniteCurve(L9, res + 100, "Progression", 1)
    # gmsh.model.mesh.setTransfiniteCurve(L10, res + 221, "Progression", 1)

    # gmsh.model.geo.synchronize()
    #
    # gmsh.option.setNumber("Mesh.RecombineAll", 2)

    # # # Defining extrusions surfaces
    # # # #2 means a surfaces, [#] specifies number of layers, recombination forms 1 cell - ie 2D mesh
    # # # extr = gmsh.model.occ.extrude([(2, Face1), (2, Face2), (2, Face3)], 0, 0, 0.1, [1], recombine=True)

    # Command synchronise() creates structures from the definitions above.
    # gmsh.model.geo.synchronize()

    # Air = gmsh.model.addPhysicalGroup(2, [Face1, Face2, Face3], name="Air")
    #
    # # Inlet
    # inlet = gmsh.model.addPhysicalGroup(1, [L1], name="Inlet")
    # # Outlet
    # outlet = gmsh.model.addPhysicalGroup(1, [L2, L3, L4, L5], name="Outlet")
    # # Wall
    # wall = gmsh.model.addPhysicalGroup(1, [L6, L7, L8], name="Wall")

    # Command synchronise() creates structures from the definitions above.
    # gmsh.model.geo.synchronize()

    # Create mesh after extrusion (connects the front and back faces)
    # gmsh.model.mesh.generate(2)

    # Generate mesh data and use ASCII2 format
    # gmsh.option.setNumber("Mesh.MshFileVersion", 2)
    # gmsh.option.setNumber("Mesh.SaveAll", 0)
    # gmsh.write("/Users/vojtechpezlar/Documents/Projects/Multi-domain Cavity/SU2/Cavity.su2")
    name = "CavityEllipse_" + str(Intersect) + ".geo_unrolled"
    gmsh.write(name)

    # # Creates  graphical user interface
    # if 'close' not in sys.argv:
    #     gmsh.fltk.run()
    #
    # # It finalizes the Gmsh API
    # gmsh.finalize()