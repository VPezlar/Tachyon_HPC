import numpy as np


def Export_VTK(filename, Rho_real, Vel_real, T_real, Rho_imag, Vel_imag, T_imag, Coords_list, Dim_arr):
    """
    Writes Legacy VTK Unstructured Grid.
    Accepts explicit Real and Imaginary parts for Rho, Velocity, and Temperature.
    Automatically calculates and writes Magnitude fields as well.
    """

    num_regions = len(Dim_arr)

    # --- 1. PRE-CALCULATE TOTALS ---
    total_points = 0
    total_cells = 0
    point_offsets = [0]

    for i in range(num_regions):
        # Dimensions passed in Dim_arr must match the ARRAY SHAPE (Number of Points)
        ny = int(Dim_arr[i, 0])
        nx = int(Dim_arr[i, 1])

        n_pts = nx * ny

        # VTK Cells (Quads) require 1 less than the number of points
        n_cells = (nx - 1) * (ny - 1)

        if n_cells < 1:
            raise ValueError(f"Region {i} dimensions ({ny}x{nx}) are too small to form a grid!")

        total_points += n_pts
        total_cells += n_cells
        point_offsets.append(total_points)

    with open(filename, 'w') as f:
        # --- 2. HEADER ---
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Combined Domain Results\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # --- 3. WRITE POINTS ---
        f.write(f"POINTS {total_points} double\n")

        for i in range(num_regions):
            X = Coords_list[0][i].flatten()
            Y = Coords_list[1][i].flatten()

            # Sanity Check: Coordinates
            ny = int(Dim_arr[i, 0])
            nx = int(Dim_arr[i, 1])
            expected_n = nx * ny

            if X.size != expected_n:
                msg = f"Region {i} Coordinate Size Mismatch! Got {X.size}, expected {expected_n} ({ny}x{nx})."
                if X.size == (ny + 1) * (nx + 1):
                    msg += "\n  -> HINT: Arrays are (N+1) but Dimensions are (N). Pass [Ny+1, Nx+1] in Dim_arr."
                raise ValueError(msg)

            # Write Coordinates
            for x, y in zip(X, Y):
                f.write(f"{x:.6f} {y:.6f} 0.0\n")

        # --- 4. WRITE CELLS ---
        f.write(f"\nCELLS {total_cells} {total_cells * 5}\n")

        for r in range(num_regions):
            ny = int(Dim_arr[r, 0])
            nx = int(Dim_arr[r, 1])
            start_idx = point_offsets[r]

            for y in range(ny - 1):
                for x in range(nx - 1):
                    p0 = start_idx + (y * nx + x)
                    p1 = p0 + 1
                    p2 = p0 + nx + 1
                    p3 = p0 + nx
                    f.write(f"4 {p0} {p1} {p2} {p3}\n")

        # --- 5. CELL TYPES ---
        f.write(f"\nCELL_TYPES {total_cells}\n")
        for _ in range(total_cells):
            f.write("9\n")  # VTK_QUAD

        # --- 6. POINT DATA ---
        f.write(f"\nPOINT_DATA {total_points}\n")

        # --- HELPER: WRITE SCALAR (Real, Imag, Mag) ---
        def write_scalar_set(name, real_list, imag_list):
            # 1. Write Real
            f.write(f"SCALARS {name}_Real double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for i in range(num_regions):
                vals = real_list[i].flatten()
                # Validate
                ny, nx = int(Dim_arr[i, 0]), int(Dim_arr[i, 1])
                if vals.size != nx * ny: raise ValueError(f"{name}_Real Region {i} size mismatch")
                for v in vals: f.write(f"{v:.6f}\n")

            # 2. Write Imag
            f.write(f"\nSCALARS {name}_Imag double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for i in range(num_regions):
                vals = imag_list[i].flatten()
                # Validate
                ny, nx = int(Dim_arr[i, 0]), int(Dim_arr[i, 1])
                if vals.size != nx * ny: raise ValueError(f"{name}_Imag Region {i} size mismatch")
                for v in vals: f.write(f"{v:.6f}\n")

            # 3. Write Magnitude (Optional but useful)
            f.write(f"\nSCALARS {name}_Mag double 1\n")
            f.write("LOOKUP_TABLE default\n")
            for i in range(num_regions):
                r_vals = real_list[i].flatten()
                i_vals = imag_list[i].flatten()
                mag_vals = np.sqrt(r_vals ** 2 + i_vals ** 2)
                for v in mag_vals: f.write(f"{v:.6f}\n")

        # --- HELPER: WRITE VECTOR (Real, Imag, Mag) ---
        def write_vector_set(name, real_list, imag_list):
            # 1. Write Real Vector
            f.write(f"\nVECTORS {name}_Real double\n")
            for i in range(num_regions):
                # Validate U component only for brevity
                ny, nx = int(Dim_arr[i, 0]), int(Dim_arr[i, 1])
                if real_list[0][i].size != nx * ny: raise ValueError(f"{name}_Real Region {i} size mismatch")

                u = real_list[0][i].flatten()
                v = real_list[1][i].flatten()
                w = real_list[2][i].flatten()
                for ui, vi, wi in zip(u, v, w):
                    f.write(f"{ui:.6f} {vi:.6f} {wi:.6f}\n")

            # 2. Write Imag Vector
            f.write(f"\nVECTORS {name}_Imag double\n")
            for i in range(num_regions):
                u = imag_list[0][i].flatten()
                v = imag_list[1][i].flatten()
                w = imag_list[2][i].flatten()
                for ui, vi, wi in zip(u, v, w):
                    f.write(f"{ui:.6f} {vi:.6f} {wi:.6f}\n")

            # 3. Write Magnitude Vector (Magnitude of each component)
            # V_mag = [ |u|, |v|, |w| ]
            f.write(f"\nVECTORS {name}_Mag double\n")
            for i in range(num_regions):
                ur, vr, wr = real_list[0][i].flatten(), real_list[1][i].flatten(), real_list[2][i].flatten()
                ui, vi, wi = imag_list[0][i].flatten(), imag_list[1][i].flatten(), imag_list[2][i].flatten()

                # Magnitude of each component independently
                mag_u = np.sqrt(ur ** 2 + ui ** 2)
                mag_v = np.sqrt(vr ** 2 + vi ** 2)
                mag_w = np.sqrt(wr ** 2 + wi ** 2)

                for mu, mv, mw in zip(mag_u, mag_v, mag_w):
                    f.write(f"{mu:.6f} {mv:.6f} {mw:.6f}\n")

        # --- WRITE ALL VARIABLES ---

        # 1. Density
        write_scalar_set("Density", Rho_real, Rho_imag)

        # 2. Temperature
        write_scalar_set("Temperature", T_real, T_imag)

        # 3. Velocity
        write_vector_set("Velocity", Vel_real, Vel_imag)


def Export_Mesh(filename, Coords_list, Dim_arr):
    """
    Export ONLY the mesh geometry with a dummy scalar variable.
    Useful for checking connectivity and grid shape without loading full flow data.
    """
    num_regions = len(Dim_arr)

    total_points = 0
    total_cells = 0
    point_offsets = [0]

    # 1. Calculate Totals
    for i in range(num_regions):
        ny = int(Dim_arr[i, 0])
        nx = int(Dim_arr[i, 1])

        n_pts = nx * ny
        n_cells = (nx - 1) * (ny - 1)

        total_points += n_pts
        total_cells += n_cells
        point_offsets.append(total_points)

    with open(filename, 'w') as f:
        # 2. Header
        f.write("# vtk DataFile Version 3.0\n")
        f.write("Mesh Geometry Check\n")
        f.write("ASCII\n")
        f.write("DATASET UNSTRUCTURED_GRID\n")

        # 3. Points
        f.write(f"POINTS {total_points} double\n")
        for i in range(num_regions):
            X = Coords_list[0][i].flatten()
            Y = Coords_list[1][i].flatten()

            # Sanity Check
            ny, nx = int(Dim_arr[i, 0]), int(Dim_arr[i, 1])
            expected_n = nx * ny
            if X.size != expected_n:
                raise ValueError(f"Region {i} mismatch: Arrays are {X.size}, Dim says {expected_n} ({ny}x{nx})")

            for x, y in zip(X, Y):
                f.write(f"{x:.6f} {y:.6f} 0.0\n")

        # 4. Cells
        f.write(f"\nCELLS {total_cells} {total_cells * 5}\n")
        for r in range(num_regions):
            ny, nx = int(Dim_arr[r, 0]), int(Dim_arr[r, 1])
            start_idx = point_offsets[r]

            for y in range(ny - 1):
                for x in range(nx - 1):
                    p0 = start_idx + (y * nx + x)
                    p1 = p0 + 1
                    p2 = p0 + nx + 1
                    p3 = p0 + nx
                    f.write(f"4 {p0} {p1} {p2} {p3}\n")

        # 5. Cell Types
        f.write(f"\nCELL_TYPES {total_cells}\n")
        for _ in range(total_cells):
            f.write("9\n")

            # 6. Dummy Data (Required so ParaView has something to color by)
        f.write(f"\nPOINT_DATA {total_points}\n")
        f.write("SCALARS MeshCheck double 1\n")
        f.write("LOOKUP_TABLE default\n")
        for _ in range(total_points):
            f.write("0.0\n")