#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import vtk
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy
from scipy.signal import find_peaks, savgol_filter

from arterial.feature_extraction.vtk_centerline_geometry.utils import (
    add_point_array,
    extract_cross_section,
    cross_section_area,
    project_to_plane_2d,
    minimum_enclosing_circle,
)


def _point_array(polydata, name):
    """
    Reads a point-data array as numpy, failing clearly when it is missing.

    Parameters
    ----------
    polydata : vtk.vtkPolyData
        Centerline polydata.
    name : str
        Name of the point-data array.

    Returns
    -------
    array : numpy.ndarray
        The array values.

    """
    array = polydata.GetPointData().GetArray(name)
    if array is None:
        raise RuntimeError(f"Centerline has no point-data array {name!r}.")
    return vtk_to_numpy(array)


def perform_radius_extraction(centerline, surface, mis_array_name="MaximumInscribedSphereRadius"):
    """
    Adds cross-section-based radius measurements as point-data arrays to a centerline
    polydata: circular-equivalent radius (CE), circumscribed-circle radius (CC), and
    ovality (MIS / CC).

    The maximum-inscribed-sphere radius (MIS) is read from the existing
    `mis_array_name` array on the centerline (computed by VMTK upstream) and is not
    duplicated. Ovality at each point is `radius_mis / radius_cc`.

    For each centerline point, the surface is cut with a plane through the point with
    the centerline tangent as normal. The connected region closest to the centerline
    point is kept, triangulated, and used to compute area (for CE) and the smallest
    enclosing circle in 2D after projecting onto the cutting plane (for CC). For
    points whose plane misses the surface (empty cross-section), CE and CC fall back
    to MIS and ovality to 1.

    Parameters
    ----------
    centerline : vtk.vtkPolyData
        Centerline polydata. Must carry a `Tangents` point-data array and a radius
        point-data array named `mis_array_name`.
    surface : vtk.vtkPolyData
        Vessel surface mesh.
    mis_array_name : str, optional
        Name of the MIS radius array on the centerline. The default is
        "MaximumInscribedSphereRadius" (the VMTK convention).

    Returns
    -------
    out : vtk.vtkPolyData
        Deep copy of `centerline` with `Radius CE`, `Radius CC`, and `Ovality` added
        as point-data arrays.

    """
    out = vtk.vtkPolyData()
    out.DeepCopy(centerline)

    radius_mis = _point_array(out, mis_array_name)
    tangents = _point_array(out, "Tangents")
    n_points = out.GetNumberOfPoints()

    radius_ce = np.empty(n_points)
    radius_cc = np.empty(n_points)

    for point_idx in range(n_points):
        center = out.GetPoints().GetPoint(point_idx)
        normal = tangents[point_idx]

        cross_section = extract_cross_section(surface, center, normal)

        if cross_section.GetNumberOfPoints() == 0:
            radius_ce[point_idx] = radius_mis[point_idx]
            radius_cc[point_idx] = radius_mis[point_idx]
            continue

        radius_ce[point_idx] = float(np.sqrt(cross_section_area(cross_section) / np.pi))

        points_3d = np.empty((cross_section.GetNumberOfPoints(), 3))
        for cs_idx in range(cross_section.GetNumberOfPoints()):
            points_3d[cs_idx] = cross_section.GetPoints().GetPoint(cs_idx)
        points_2d = project_to_plane_2d(points_3d, center, normal)
        _, _, radius_cc[point_idx] = minimum_enclosing_circle(points_2d)

    ovality = np.divide(radius_mis, radius_cc, out=np.ones_like(radius_mis, dtype=float), where=radius_cc > 0)

    add_point_array(out, radius_ce, "Radius CE")
    add_point_array(out, radius_cc, "Radius CC")
    add_point_array(out, ovality, "Ovality")

    return out


def perform_curvature_extraction(centerline, savgol_window_length=50, savgol_polyorder=3):
    """
    Adds curvature-related point-data arrays to a centerline polydata: `Curvature`
    and `Torsion` (computed from the Frenet-Serret frame stored on the centerline as
    `Tangents`, `Normals`, `Binormals`), `Filtered curvature` (Savitzky-Golay smoothed
    curvature), and `Distance from origin` (cumulative arc length from the first
    point).

    Parameters
    ----------
    centerline : vtk.vtkPolyData
        Centerline polydata with `Tangents`, `Normals`, and `Binormals` point-data
        arrays (as produced by VMTK / Slicer centerline computation).
    savgol_window_length : int, optional
        Window length for Savitzky-Golay smoothing of the curvature signal. The
        default is 50. Clipped to the number of centerline points if larger.
    savgol_polyorder : int, optional
        Polynomial order for Savitzky-Golay smoothing. The default is 3.

    Returns
    -------
    out : vtk.vtkPolyData
        Deep copy of `centerline` with `Curvature`, `Torsion`, `Filtered curvature`,
        and `Distance from origin` added as point-data arrays.

    """
    out = vtk.vtkPolyData()
    out.DeepCopy(centerline)

    n_points = out.GetNumberOfPoints()
    coordinates = np.empty((n_points, 3))
    for point_idx in range(n_points):
        coordinates[point_idx] = out.GetPoints().GetPoint(point_idx)

    distance_from_origin = np.zeros(n_points)
    distance_from_origin[1:] = np.cumsum(np.linalg.norm(np.diff(coordinates, axis=0), axis=1))

    tangents = _point_array(out, "Tangents")
    normals = _point_array(out, "Normals")
    binormals = _point_array(out, "Binormals")

    curvature = np.linalg.norm(np.gradient(tangents, axis=0), axis=1)
    torsion = (-np.gradient(binormals, axis=0) * normals).sum(axis=1)

    filtered_curvature = savgol_filter(curvature, window_length=savgol_window_length, polyorder=savgol_polyorder, mode="nearest")

    # Angle of curvature: dihedral angle (degrees) between consecutive tangent
    # vectors. Attached to the receiving point so cumsum reads as total turning
    # from the start of the centerline up to point i.
    angle_of_curvature = np.zeros(n_points)
    cos_theta = np.clip(np.einsum("ij,ij->i", tangents[:-1], tangents[1:]), -1.0, 1.0)
    angle_of_curvature[1:] = np.degrees(np.arccos(cos_theta))
    cumulative_angle_of_curvature = np.cumsum(angle_of_curvature)

    add_point_array(out, curvature, "Curvature")
    add_point_array(out, torsion, "Torsion")
    add_point_array(out, distance_from_origin, "Distance from origin")
    add_point_array(out, filtered_curvature, "Filtered curvature")
    add_point_array(out, angle_of_curvature, "Angle of curvature")
    add_point_array(out, cumulative_angle_of_curvature, "Cumulative angle of curvature")

    return out


def perform_curve_id_extraction(centerline, peak_height=0.030, peak_width=10):
    """
    Segments a centerline into successive turns ("curves") by detecting peaks in
    the `Filtered curvature` point-data array and placing curve boundaries at the
    curvature minimum between each consecutive pair of peaks. Each turn is assigned
    an integer identifier starting at 0 from the proximal end and incrementing by
    one at each boundary; the result is added as a `CurveIds` integer point-data
    array.

    The centerline must already carry a `Filtered curvature` array (produced by
    `perform_curvature_extraction`). A `ValueError` is raised if it is missing,
    rather than silently re-computing it: this keeps each step single-purpose and
    lets callers control which arrays are written.

    Parameters
    ----------
    centerline : vtk.vtkPolyData
        Centerline polydata with a `Filtered curvature` point-data array.
    peak_height : float, optional
        Minimum peak height passed to `scipy.signal.find_peaks`. Tuned for
        ICA-scale vessels at 2 mm sampling; vessels at finer/coarser sampling or
        of different calibre will likely need adjustment. The default is 0.030.
    peak_width : int, optional
        Minimum peak width in samples, passed to `scipy.signal.find_peaks`. The
        default is 10.

    Returns
    -------
    out : vtk.vtkPolyData
        Deep copy of `centerline` with `CurveIds` added as an int point-data array.

    """
    out = vtk.vtkPolyData()
    out.DeepCopy(centerline)

    filtered_curvature_array = out.GetPointData().GetArray("Filtered curvature")
    if filtered_curvature_array is None:
        raise ValueError("`Filtered curvature` point-data array not found on centerline; run `perform_curvature_extraction` first.")
    filtered_curvature = vtk_to_numpy(filtered_curvature_array)

    peaks, _ = find_peaks(filtered_curvature, height=peak_height, width=peak_width)
    separation_points = [
        peaks[idx - 1] + int(np.argmin(filtered_curvature[peaks[idx - 1]:peaks[idx]]))
        for idx in range(1, len(peaks))
    ]

    curve_ids = np.zeros(len(filtered_curvature), dtype=int)
    for curve_id, sep in enumerate(separation_points, start=1):
        curve_ids[sep:] = curve_id

    add_point_array(out, curve_ids, "CurveIds")
    return out
