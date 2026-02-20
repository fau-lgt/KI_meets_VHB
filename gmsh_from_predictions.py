"""Generate the 2D microstructured topology from predictions_*.txt using Gmsh.

Requirements:
  pip install gmsh numpy

Usage:
  python gmsh_from_predictions_inp.py --pred predictions_5x15.txt --out topo.inp

Notes:
  - Uses the OpenCASCADE (occ) kernel and boolean cuts.
  - Calls removeAllDuplicates() at the end to merge coincident interfaces.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List, Tuple
from pathlib import Path
import numpy as np
import gmsh

DimTag = Tuple[int, int]  # (dim, tag)


def scale_to_arbitrary_range(x01: float, r: Tuple[float, float]) -> float:
    if not (0.0 <= x01 <= 1.0):
        raise ValueError("Input number must be in the range [0, 1].")
    a, b = r
    if not (a < b):
        raise ValueError("Invalid range: range[0] must be < range[1].")
    return (b - a) * x01 + a


@dataclass(frozen=True)
class Cell:
    density01: float
    mtype: int
    x: float
    y: float


def add_triangle_surface(occ, p1, p2, p3, lc=0.0) -> int:
    """Create a planar triangular surface in OCC and return its surface tag."""
    # OCC entities are created with numeric tags; duplicates will be merged later.
    a = occ.addPoint(p1[0], p1[1], 0.0, lc)
    b = occ.addPoint(p2[0], p2[1], 0.0, lc)
    c = occ.addPoint(p3[0], p3[1], 0.0, lc)

    l1 = occ.addLine(a, b)
    l2 = occ.addLine(b, c)
    l3 = occ.addLine(c, a)

    cl = occ.addCurveLoop([l1, l2, l3])
    s = occ.addPlaneSurface([cl])
    return s


def create_square_microstructure(occ, size, pos, t, scale_factor) -> List[DimTag]:
    """Outer rectangle minus inner rectangle (frame)."""
    if t < 0:
        raise ValueError("Thickness cannot be negative")

    if t <= 0.05 / scale_factor:
        return []

    x, y = pos
    dx, dy = size

    outer = occ.addRectangle(x, y, 0.0, dx, dy)

    # Inner void
    if t < 0.45 / scale_factor:
        inner = occ.addRectangle(x + t, y + t, 0.0, dx - 2 * t, dy - 2 * t)
        out, _ = occ.cut([(2, outer)], [(2, inner)], removeObject=True, removeTool=True)
        return out

    return [(2, outer)]


def create_xbox_microstructure(occ, size, pos, t, scale_factor) -> List[DimTag]:
    """Outer rectangle minus 4 triangular cutouts (left/right/top/bottom)."""
    if t < 0:
        raise ValueError("Thickness cannot be negative")

    if t <= 0.03 / scale_factor:
        return []

    x, y = pos
    dx, dy = size

    outer = occ.addRectangle(x, y, 0.0, dx, dy)

    if t >= 0.28 / scale_factor:
        return [(2, outer)]

    tris = []

    # Left triangle
    p1 = (x + 1.0 * t, y + 1.75 * t)
    p2 = (x + 1.0 * t, y + dy - 1.75 * t)
    p3 = (x + dx / 2.0 - 0.75 * t, y + dy / 2.0)
    tris.append(add_triangle_surface(occ, p1, p2, p3))

    # Right triangle
    p1 = (x + dx - 1.0 * t, y + 1.75 * t)
    p2 = (x + dx - 1.0 * t, y + dy - 1.75 * t)
    p3 = (x + dx / 2.0 + 0.75 * t, y + dy / 2.0)
    tris.append(add_triangle_surface(occ, p1, p2, p3))

    # Bottom triangle
    p1 = (x + 1.75 * t, y + 1.0 * t)
    p2 = (x + dx / 2.0, y + dy / 2.0 - 0.75 * t)
    p3 = (x + dx - 1.75 * t, y + 1.0 * t)
    tris.append(add_triangle_surface(occ, p1, p2, p3))

    # Top triangle
    p1 = (x + 1.75 * t, y + dy - 1.0 * t)
    p2 = (x + dx - 1.75 * t, y + dy - 1.0 * t)
    p3 = (x + dx / 2.0, y + dy / 2.0 + 0.75 * t)
    tris.append(add_triangle_surface(occ, p1, p2, p3))

    out, _ = occ.cut([(2, outer)], [(2, s) for s in tris], removeObject=True, removeTool=True)
    return out


def create_xpbox_microstructure(occ, size, pos, t, scale_factor) -> List[DimTag]:
    """Outer rectangle minus 8 triangular cutouts (the 'xpbox' pattern)."""
    if t < 0:
        raise ValueError("Thickness cannot be negative")

    if t <= 0.03 / scale_factor:
        return []

    x, y = pos
    dx, dy = size

    outer = occ.addRectangle(x, y, 0.0, dx, dy)

    if t >= 0.21 / scale_factor:
        return [(2, outer)]

    tris = []

    # Helper aliases
    cx = x + dx / 2.0
    cy = y + dy / 2.0

    # Left-bottom
    p1 = (x + 1.0 * t, y + 1.75 * t)
    p2 = (x + 1.0 * t, cy - 0.5 * t)
    p3 = (cx - 1.25 * t, cy - 0.5 * t)
    tris.append(add_triangle_surface(occ, p1, p2, p3))

    # Left-top
    p1 = (x + 1.0 * t, cy + 0.5 * t)
    p2 = (x + 1.0 * t, y + dy - 1.75 * t)
    p3 = (cx - 1.25 * t, cy + 0.5 * t)
    tris.append(add_triangle_surface(occ, p1, p2, p3))

    # Right-bottom
    p1 = (x + dx - 1.0 * t, y + 1.75 * t)
    p2 = (x + dx - 1.0 * t, cy - 0.5 * t)
    p3 = (cx + 1.25 * t, cy - 0.5 * t)
    tris.append(add_triangle_surface(occ, p1, p2, p3))

    # Right-top
    p1 = (x + dx - 1.0 * t, cy + 0.5 * t)
    p2 = (cx + 1.25 * t, cy + 0.5 * t)
    p3 = (x + dx - 1.0 * t, y + dy - 1.75 * t)
    tris.append(add_triangle_surface(occ, p1, p2, p3))

    # Bottom-left
    p1 = (x + 1.75 * t, y + 1.0 * t)
    p2 = (cx - 0.5 * t, cy - 1.25 * t)
    p3 = (cx - 0.5 * t, y + 1.0 * t)
    tris.append(add_triangle_surface(occ, p1, p2, p3))

    # Bottom-right
    p1 = (cx + 0.5 * t, y + 1.0 * t)
    p2 = (x + dx - 1.75 * t, y + 1.0 * t)
    p3 = (cx + 0.5 * t, cy - 1.25 * t)
    tris.append(add_triangle_surface(occ, p1, p2, p3))

    # Top-left
    p1 = (x + 1.75 * t, y + dy - 1.0 * t)
    p2 = (cx - 0.5 * t, y + dy - 1.0 * t)
    p3 = (cx - 0.5 * t, cy + 1.25 * t)
    tris.append(add_triangle_surface(occ, p1, p2, p3))

    # Top-right
    p1 = (cx + 0.5 * t, y + dy - 1.0 * t)
    p2 = (cx + 0.5 * t, cy + 1.25 * t)
    p3 = (x + dx - 1.75 * t, y + dy - 1.0 * t)
    tris.append(add_triangle_surface(occ, p1, p2, p3))

    out, _ = occ.cut([(2, outer)], [(2, s) for s in tris], removeObject=True, removeTool=True)
    return out


def read_predictions(path: str) -> List[Cell]:
    """Read tab-separated file with columns: density(%), microstructure_type, x, y."""
    # Be tolerant: allow whitespace separators too.
    data = np.loadtxt(path, delimiter=None)
    if data.ndim == 1:
        data = data[None, :]

    density = data[:, 0] / 100.0
    mtype = data[:, 1].astype(int)
    px = data[:, 2]
    py = data[:, 3]

    cells = [Cell(float(d), int(t), float(x), float(y)) for d, t, x, y in zip(density, mtype, px, py)]
    return cells


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", help="predictions_*.txt path", default="T:/4_FUNDING/VHB meets KI/GM-TOuNN Topology Optimization/Matlab_Comsol reconstruction/Python_GMSH/Test_problem_2/60x30/Test 1, vf = 30/density_unity_cell_x_y.txt")
    ap.add_argument("--out", default="topology_problem1.inp", help="output file (.inp for PrePoMax/CalculiX/Abaqus, or .msh/.unv)")
    ap.add_argument("--geo", default=None, help="optional output geometry file (.brep or .step)")
    ap.add_argument("--scale", type=float, default=1 / 0.03, help="scale_factor (MATLAB uses 1/0.03)")
    ap.add_argument("--lc", type=float, default=0.003, help="optional target mesh size (0 lets Gmsh choose)")
    ap.add_argument("--show", action="store_true", help="open the Gmsh GUI")
    args = ap.parse_args()

    # Output: default to Abaqus/CalculiX .inp (importable in PrePoMax).
    out_path = args.out
    if not Path(out_path).suffix:
        out_path = out_path + ".inp"
    out_ext = Path(out_path).suffix.lower()
    if out_ext not in {".inp", ".msh", ".unv"}:
        raise ValueError(f"Unsupported output extension: {out_ext}. Use .inp, .msh or .unv")

    cells = read_predictions(args.pred)

    # MATLAB uses cube_size = [1,1] and divides by scale_factor
    cell_size = (1.0 / args.scale, 1.0 / args.scale)

    gmsh.initialize()
    gmsh.model.add("topology")
    occ = gmsh.model.occ

    all_surfs: List[DimTag] = []

    for c in cells:
        pos = (c.x / args.scale, c.y / args.scale)

        if c.mtype == 0:
            t = scale_to_arbitrary_range(c.density01, (0.0, 0.45 / args.scale))
            out = create_square_microstructure(occ, cell_size, pos, t, args.scale)
        elif c.mtype == 1:
            t = scale_to_arbitrary_range(c.density01, (0.0, 0.20 / args.scale))
            out = create_xbox_microstructure(occ, cell_size, pos, t, args.scale)
        elif c.mtype == 2:
            t = scale_to_arbitrary_range(c.density01, (0.0, 0.15 / args.scale))
            out = create_xpbox_microstructure(occ, cell_size, pos, t, args.scale)
        else:
            raise ValueError(f"Unknown microstructure_type={c.mtype}")

        # Keep only surfaces
        all_surfs.extend([(d, t) for (d, t) in out if d == 2])

    occ.synchronize()

    # Merge coincident interfaces so the final mesh is conformal across cell boundaries.
    # (For complex models, BooleanFragments can also be used.)
    occ.removeAllDuplicates()
    occ.synchronize()

    # Physical group for the whole solid domain
    surf_tags = [tag for (dim, tag) in gmsh.model.getEntities(2)]
    if surf_tags:
        pg = gmsh.model.addPhysicalGroup(2, surf_tags)
        gmsh.model.setPhysicalName(2, pg, "solid")

    if args.geo:
        gmsh.write(args.geo)

    #if args.lc > 0:
    gmsh.option.setNumber("Mesh.MeshSizeMin", args.lc)
    gmsh.option.setNumber("Mesh.MeshSizeMax", args.lc)

    gmsh.model.mesh.generate(2)
    # Write mesh to requested format based on file extension (.inp/.msh/.unv)
    if out_ext == ".inp":
        # Make sure only entities in physical groups are exported (recommended).
        gmsh.option.setNumber("Mesh.SaveAll", 0)
        # Abaqus/CalculiX input files are ASCII.
        gmsh.option.setNumber("Mesh.Binary", 0)
    gmsh.write(out_path)

    if args.show:
        gmsh.fltk.run()

    gmsh.finalize()


if __name__ == "__main__":
    main()
