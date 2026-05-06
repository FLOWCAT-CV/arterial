#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import vtk
import numpy as np
import nibabel as nib

from scipy.spatial import ConvexHull, cKDTree
from scipy.spatial.qhull import QhullError
from vtk.util.numpy_support import numpy_to_vtk, vtk_to_numpy


def add_point_array(polydata, array, name):
    """
    Adds a numpy array as a named point-data array to a vtkPolyData (in place).

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        Target polydata.
    array : np.ndarray
        1-D or 2-D numpy array with one row per point.
    name : str
        Name to assign to the new point-data array.

    """
    vtk_array = numpy_to_vtk(np.ascontiguousarray(array))
    vtk_array.SetName(name)
    polydata.GetPointData().AddArray(vtk_array)


def add_string_point_array(polydata, strings, name):
    """
    Adds a Python list of strings as a named `vtkStringArray` point-data array to
    a vtkPolyData (in place). Used for categorical labels (e.g. vessel type names)
    that are not amenable to numeric storage.

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        Target polydata.
    strings : sequence of str
        One string per point.
    name : str
        Name to assign to the new point-data array.

    """
    vtk_array = vtk.vtkStringArray()
    vtk_array.SetName(name)
    vtk_array.SetNumberOfValues(len(strings))
    for i, s in enumerate(strings):
        vtk_array.SetValue(i, str(s))
    polydata.GetPointData().AddArray(vtk_array)


def extract_cross_section(surface, center, normal):
    """
    Cuts the surface with a plane defined by center and normal, keeps the connected
    region closest to center, and returns the triangulated cross-section polydata.

    Parameters
    ----------
    surface : vtk.vtkPolyData
        Closed (or reasonably closed) vessel surface mesh.
    center : array-like, shape (3,)
        Plane origin (typically a centerline point).
    normal : array-like, shape (3,)
        Plane normal (typically the centerline tangent at `center`).

    Returns
    -------
    cross_section : vtk.vtkPolyData
        Triangulated cross-section polydata. Empty if the cutting plane does not
        intersect the surface near `center`.

    """
    plane = vtk.vtkPlane()
    plane.SetOrigin(center[0], center[1], center[2])
    plane.SetNormal(normal[0], normal[1], normal[2])

    cutter = vtk.vtkCutter()
    cutter.SetInputData(surface)
    cutter.SetCutFunction(plane)
    cutter.Update()

    connectivity = vtk.vtkConnectivityFilter()
    connectivity.SetInputData(cutter.GetOutput())
    connectivity.SetClosestPoint(center[0], center[1], center[2])
    connectivity.SetExtractionModeToClosestPointRegion()
    connectivity.Update()

    triangulator = vtk.vtkContourTriangulator()
    triangulator.SetInputData(connectivity.GetPolyDataOutput())
    triangulator.Update()

    return triangulator.GetOutput()


def cross_section_area(cross_section):
    """
    Computes the surface area of a triangulated cross-section polydata.

    """
    mass = vtk.vtkMassProperties()
    mass.SetInputData(cross_section)
    mass.Update()
    return mass.GetSurfaceArea()


def plane_basis(normal):
    """
    Returns two orthonormal vectors spanning the plane perpendicular to `normal`.

    """
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    ref = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    u = np.cross(n, ref)
    u = u / np.linalg.norm(u)
    v = np.cross(n, u)
    return u, v


def project_to_plane_2d(points_3d, origin, normal):
    """
    Projects 3D points onto a 2D coordinate system in the plane defined by origin
    and normal.

    """
    u, v = plane_basis(normal)
    rel = np.asarray(points_3d, dtype=float) - np.asarray(origin, dtype=float)
    return np.column_stack([rel @ u, rel @ v])


def _circle_from_two(p1, p2):
    cx = 0.5 * (p1[0] + p2[0])
    cy = 0.5 * (p1[1] + p2[1])
    r = 0.5 * float(np.hypot(p2[0] - p1[0], p2[1] - p1[1]))
    return cx, cy, r


def _circle_from_three(p1, p2, p3):
    ax, ay = p1; bx, by = p2; cx, cy = p3
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-12:
        # Collinear: fall back to the largest pairwise diameter circle
        return max(
            (_circle_from_two(p1, p2), _circle_from_two(p1, p3), _circle_from_two(p2, p3)),
            key=lambda c: c[2],
        )
    a2, b2, c2 = ax * ax + ay * ay, bx * bx + by * by, cx * cx + cy * cy
    ux = (a2 * (by - cy) + b2 * (cy - ay) + c2 * (ay - by)) / d
    uy = (a2 * (cx - bx) + b2 * (ax - cx) + c2 * (bx - ax)) / d
    return ux, uy, float(np.hypot(ax - ux, ay - uy))


def _circle_from_boundary(boundary):
    if len(boundary) == 0:
        return 0.0, 0.0, 0.0
    if len(boundary) == 1:
        return float(boundary[0][0]), float(boundary[0][1]), 0.0
    if len(boundary) == 2:
        return _circle_from_two(boundary[0], boundary[1])
    return _circle_from_three(boundary[0], boundary[1], boundary[2])


def _in_circle(point, circle, eps=1e-9):
    cx, cy, r = circle
    return np.hypot(point[0] - cx, point[1] - cy) <= r + eps


def _welzl(points, boundary, n):
    if n == 0 or len(boundary) == 3:
        return _circle_from_boundary(boundary)
    p = points[n - 1]
    circle = _welzl(points, boundary, n - 1)
    if _in_circle(p, circle):
        return circle
    return _welzl(points, boundary + [p], n - 1)


def lpi_corner_coordinates(affine, image_shape):
    """
    Returns the real-world coordinates of the LPI (left-posterior-inferior) corner
    of an image, used by arterial as the origin of its graph coordinate system.
    Graph node positions in arterial pickles equal `vtk_native_coord - lpi_corner`,
    so this function provides the offset needed to convert arterial graph coordinates
    back to native NIfTI/VTK space.

    Parameters
    ----------
    affine : np.ndarray, shape (4, 4)
        Affine matrix of the NIfTI image.
    image_shape : tuple of int
        Shape of the NIfTI image (at least the first three dimensions).

    Returns
    -------
    lpi_corner : np.ndarray, shape (3,)
        Real-world coordinates of the LPI corner.

    """
    orientation = nib.aff2axcodes(affine)
    if orientation == ("R", "A", "S"):
        lpi_voxel = np.array([0, 0, 0])
    elif orientation == ("L", "A", "S"):
        lpi_voxel = np.array([image_shape[0] - 1, 0, 0])
    elif orientation == ("L", "P", "S"):
        lpi_voxel = np.array([image_shape[0] - 1, image_shape[1] - 1, 0])
    else:
        raise ValueError(f"Unsupported image orientation {orientation}; arterial supports RAS, LAS, LPS.")
    return np.dot(affine, np.append(lpi_voxel, 1))[:3]


def walk_chain(graph):
    """
    Returns the ordered node sequence of a linear-chain graph by walking from one
    endpoint to the other. Used to recover centerline ordering from arterial's
    single-segment graphs, whose node ids are typically not contiguous.

    Falls back to `list(graph.nodes)` for graphs without degree-1 endpoints (closed
    loops or single-node graphs).

    """
    endpoints = [n for n in graph.nodes if graph.degree(n) == 1]
    if not endpoints:
        return list(graph.nodes)
    start = endpoints[0]
    ordered = [start]
    visited = {start}
    current = start
    while True:
        nexts = [nb for nb in graph.neighbors(current) if nb not in visited]
        if not nexts:
            break
        current = nexts[0]
        ordered.append(current)
        visited.add(current)
    return ordered


def compute_frenet_frame(coordinates):
    """
    Computes Tangent / Normal / Binormal arrays along a polyline using a
    rotation-minimizing frame (double-reflection method, Wang et al. 2008). The
    resulting frame is C1-continuous and stable across straight sections, which is
    what downstream curvature/torsion code expects.

    Parameters
    ----------
    coordinates : np.ndarray, shape (n, 3)
        Ordered polyline points.

    Returns
    -------
    tangents, normals, binormals : np.ndarray, each shape (n, 3)

    """
    n = len(coordinates)
    tangents = np.gradient(coordinates, axis=0)
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    tangents = tangents / norms

    normals = np.empty_like(tangents)
    binormals = np.empty_like(tangents)

    ref = np.array([1.0, 0.0, 0.0]) if abs(tangents[0, 0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    n0 = np.cross(tangents[0], ref)
    n0 = n0 / np.linalg.norm(n0)
    normals[0] = n0
    binormals[0] = np.cross(tangents[0], normals[0])

    for i in range(n - 1):
        v1 = coordinates[i + 1] - coordinates[i]
        c1 = float(v1 @ v1)
        if c1 < 1e-20:
            normals[i + 1] = normals[i]
            binormals[i + 1] = binormals[i]
            continue
        n_l = normals[i] - (2.0 / c1) * (v1 @ normals[i]) * v1
        t_l = tangents[i] - (2.0 / c1) * (v1 @ tangents[i]) * v1
        v2 = tangents[i + 1] - t_l
        c2 = float(v2 @ v2)
        if c2 < 1e-20:
            normals[i + 1] = n_l
        else:
            normals[i + 1] = n_l - (2.0 / c2) * (v2 @ n_l) * v2
        normals[i + 1] = normals[i + 1] / max(np.linalg.norm(normals[i + 1]), 1e-12)
        binormals[i + 1] = np.cross(tangents[i + 1], normals[i + 1])

    return tangents, normals, binormals


def branch_model_blanking_lookup(branch_model):
    """
    Builds parallel arrays of branch-model point coordinates (in native space) and
    the Blanking cell value associated with each, suitable for nearest-neighbour
    lookup. Mirrors arterial's `featurize_node` blanking pool but operates in native
    space (no LPI subtraction).

    Parameters
    ----------
    branch_model : vtk.vtkPolyData
        VMTK branch model with a `Blanking` cell-data array.

    Returns
    -------
    coordinates : np.ndarray, shape (n, 3)
    blanking : np.ndarray, shape (n,), int

    """
    blanking_cell_array = vtk_to_numpy(branch_model.GetCellData().GetArray("Blanking"))
    n_total = sum(branch_model.GetCell(c).GetNumberOfPoints() for c in range(branch_model.GetNumberOfCells()))
    coordinates = np.empty((n_total, 3))
    blanking = np.empty(n_total, dtype=int)
    idx = 0
    for c in range(branch_model.GetNumberOfCells()):
        cell = branch_model.GetCell(c)
        cell_blanking = int(blanking_cell_array[c])
        for p in range(cell.GetNumberOfPoints()):
            coordinates[idx] = cell.GetPoints().GetPoint(p)
            blanking[idx] = cell_blanking
            idx += 1
    return coordinates, blanking


def pickle_to_vtk(graph, affine, image_shape, branch_model=None, mis_array_name="MaximumInscribedSphereRadius", compute_frenet=True):
    """
    Converts a featurized single-segment networkx graph (from arterial's
    `single_segments/` output) to a `vtkPolyData` centerline in native NIfTI/VTK
    coordinates. Applies the inverse of arterial's LPI-corner translation so the
    result aligns with the original CTA and segmentation surface in real-world space.

    The graph is walked from one of its endpoints to recover centerline ordering
    (node ids in single-segment pickles are typically non-contiguous). Node positions
    are written as `vtkPoints` connected by a single polyline cell. The MIS radius
    is read from each node's `radius` attribute and stored under `mis_array_name`.
    `VesselType` (int) and `VesselTypeName` (vtkStringArray) are read from each
    node's `vessel_type` / `vessel_type_name` attributes — useful because a single
    segment's path typically traverses several vessel labels (e.g. AA → LCCA → LICA).
    If `compute_frenet=True`, `Tangents`, `Normals`, and `Binormals` are computed
    from the polyline geometry and stored as point-data arrays so the result is
    immediately consumable by `perform_radius_extraction` and
    `perform_curvature_extraction`.

    Parameters
    ----------
    graph : networkx.Graph
        Single-segment graph from arterial's `single_segments/` directory.
    affine : np.ndarray, shape (4, 4)
        Affine matrix of the CTA NIfTI.
    image_shape : tuple of int
        Shape of the CTA volume.
    branch_model : vtk.vtkPolyData, optional
        VMTK branch model with a `Blanking` cell-data array. If provided, blanking
        is derived directly from the branch model via nearest-neighbour lookup
        (the source of truth). If None, blanking falls back to each node's
        `features femoral["blanking"]` value, which may be stale or zero if the
        upstream `extract_local_features` ran without a branch model.
    mis_array_name : str, optional
        Name to assign to the MIS radius point-data array. The default is
        "MaximumInscribedSphereRadius" (the VMTK convention).
    compute_frenet : bool, optional
        Whether to compute and store `Tangents`, `Normals`, `Binormals` arrays via
        a rotation-minimizing frame. The default is True.

    Returns
    -------
    centerline : vtk.vtkPolyData
        Centerline polydata in native space with `mis_array_name` (and optionally
        Frenet-Serret) point-data arrays.

    """
    ordered_nodes = walk_chain(graph)
    lpi_corner = lpi_corner_coordinates(affine, image_shape)

    coordinates = np.array([np.asarray(graph.nodes[n]["pos"]) + lpi_corner for n in ordered_nodes])
    radius = np.array([float(graph.nodes[n]["radius"]) for n in ordered_nodes])
    vessel_type = np.array([int(graph.nodes[n]["vessel_type"]) for n in ordered_nodes])
    vessel_type_name = [str(graph.nodes[n]["vessel_type_name"]) for n in ordered_nodes]
    if branch_model is not None:
        bm_coords, bm_blanking = branch_model_blanking_lookup(branch_model)
        _, nearest = cKDTree(bm_coords).query(coordinates)
        blanking = bm_blanking[nearest]
    else:
        blanking = np.array([int(graph.nodes[n]["features femoral"]["blanking"]) for n in ordered_nodes])

    points = vtk.vtkPoints()
    for c in coordinates:
        points.InsertNextPoint(c[0], c[1], c[2])

    polyline = vtk.vtkPolyLine()
    polyline.GetPointIds().SetNumberOfIds(len(ordered_nodes))
    for i in range(len(ordered_nodes)):
        polyline.GetPointIds().SetId(i, i)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(polyline)

    centerline = vtk.vtkPolyData()
    centerline.SetPoints(points)
    centerline.SetLines(cells)
    add_point_array(centerline, radius, mis_array_name)
    add_point_array(centerline, blanking, "Blanking")
    add_point_array(centerline, vessel_type, "VesselType")
    add_string_point_array(centerline, vessel_type_name, "VesselTypeName")

    if compute_frenet:
        tangents, normals, binormals = compute_frenet_frame(coordinates)
        add_point_array(centerline, tangents, "Tangents")
        add_point_array(centerline, normals, "Normals")
        add_point_array(centerline, binormals, "Binormals")

    return centerline


def minimum_enclosing_circle(points_2d, seed=0):
    """
    Smallest circle enclosing a set of 2D points (Welzl's algorithm).

    Reduces the input to its convex-hull vertices first (the optimal circle is
    determined by hull points), then runs Welzl on a randomized permutation for
    expected linear time.

    Parameters
    ----------
    points_2d : array-like, shape (n, 2)
        Points in 2D.
    seed : int, optional
        Seed for the permutation, for reproducibility. The default is 0.

    Returns
    -------
    (cx, cy, radius) : tuple of floats

    """
    pts = np.asarray(points_2d, dtype=float)
    if len(pts) == 0:
        return 0.0, 0.0, 0.0
    if len(pts) >= 3:
        try:
            pts = pts[ConvexHull(pts).vertices]
        except QhullError:
            pass  # collinear / degenerate input — fall through to raw points
    rng = np.random.default_rng(seed)
    pts = pts[rng.permutation(len(pts))]
    return _welzl([tuple(p) for p in pts], [], len(pts))
