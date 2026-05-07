#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import vtk
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import PchipInterpolator

from arterial.feature_extraction.vtk_centerline_geometry.utils import add_point_array


def perform_carotid_analysis(
    centerline,
    *,
    mis_array_name="MaximumInscribedSphereRadius",
    n_bif_mask_nodes=20,
    bulb_threshold_mm=0.5,
    bulb_expand_nodes=1,
):
    """
    Detects the carotid bulb on a CCA→ICA centerline polydata and adds the
    intermediate signals as point-data arrays.

    The algorithm proceeds in four steps:

    1. **Trim leading/trailing blanking.** Drops only the leading and trailing
       runs of ``Blanking == 1`` (which mark the edges of upstream / downstream
       segments that are not part of the CCA→ICA path of interest). Middle
       blanking points (the carotid bifurcation neighbourhood) are preserved.
    2. **Find the CCA→ICA transition.** Identified as the first centerline
       point whose ``VesselTypeName`` contains the substring ``"ICA"`` (matches
       both ``LICA`` and ``RICA``).
    3. **Interpolate the radius across the bifurcation.** Masks
       ``n_bif_mask_nodes`` centerline nodes around the transition (split as
       ``n // 2`` proximal + remainder starting at the transition itself) and
       fits a PCHIP curve through the surviving non-masked points, evaluated
       at every centerline arclength.
    4. **Classify bulb nodes.** Threshold-positive where
       ``|raw_radius - r_interp| > bulb_threshold_mm``. Each contiguous run of
       positives is then symmetrically extended by ``bulb_expand_nodes`` on
       each side via a 1-D binary dilation (clamped to the array boundaries).

    Returns a deep copy of the input polydata with five new point-data arrays
    added (see below). Trimmed-out points (leading/trailing blanking) carry
    ``NaN`` in the radius/diff arrays so the centerline geometry stays intact
    for ParaView; binary masks carry ``0`` at trimmed-out positions.

    Parameters
    ----------
    centerline : vtk.vtkPolyData
        Per-vessel centerline polydata spanning CCA → ICA. Must carry
        ``Blanking`` (int 0/1), ``VesselTypeName`` (vtkStringArray),
        ``Distance from origin`` (float, monotone arclength in mm), and the
        radius array named by ``mis_array_name``.
    mis_array_name : str, optional
        Name of the radius array on the centerline. The default is
        ``"MaximumInscribedSphereRadius"`` (the VMTK convention).
    n_bif_mask_nodes : int, optional
        Total number of centerline nodes to mask around the CCA→ICA
        transition before interpolating the radius. The default is 20.
    bulb_threshold_mm : float, optional
        A centerline node is labelled bulb when ``|raw - interp| >
        bulb_threshold_mm``. The default is 0.5.
    bulb_expand_nodes : int, optional
        Symmetric expansion of every threshold-positive run of bulb nodes,
        in nodes per side. The default is 1.

    Returns
    -------
    out : vtk.vtkPolyData
        Deep copy of ``centerline`` with the following point-data arrays added:

        - ``KeepAfterBlankingTrim`` (int 0/1): which original nodes survived
          the leading/trailing blanking trim.
        - ``Radius MIS interp`` (float, NaN where trimmed): the PCHIP baseline
          radius across the bifurcation-mask window.
        - ``Radius MIS diff (raw - interp)`` (float, NaN where trimmed): the
          residual ``raw - interp``.
        - ``BifurcationMask`` (int 0/1, 0 where trimmed): the
          ``n_bif_mask_nodes``-wide window around the CCA→ICA transition.
        - ``BulbMask`` (int 0/1, 0 where trimmed): the final bulb
          classification.

    Raises
    ------
    RuntimeError
        If no CCA→ICA transition is found (no ICA-labelled points, the
        centerline is entirely ICA, or the centerline starts on ICA so there
        are no CCA points proximal to interpolate from).
    ValueError
        If too few centerline nodes survive the bifurcation mask to fit a
        PCHIP curve through.

    """
    out = vtk.vtkPolyData()
    out.DeepCopy(centerline)

    n_full = out.GetNumberOfPoints()

    blanking_full = vtk_to_numpy(out.GetPointData().GetArray("Blanking")).astype(int)
    s_full = vtk_to_numpy(out.GetPointData().GetArray("Distance from origin")).astype(float)
    r_full = vtk_to_numpy(out.GetPointData().GetArray(mis_array_name)).astype(float)

    vt_name_array = out.GetPointData().GetAbstractArray("VesselTypeName")
    if vt_name_array is None:
        raise RuntimeError(
            "VesselTypeName point-data array is required for carotid analysis "
            "but was not found on the input centerline."
        )
    vt_name_full = [vt_name_array.GetValue(i) for i in range(n_full)]

    keep = _leading_trailing_blanking_keep_mask(blanking_full)
    s = s_full[keep]
    r = r_full[keep]
    vt_name = [vt_name_full[i] for i in np.where(keep)[0]]

    transition_idx = _find_cca_ica_transition(vt_name)

    half = n_bif_mask_nodes // 2
    lo = max(0, transition_idx - half)
    hi = min(s.size, transition_idx + (n_bif_mask_nodes - half))

    bif_mask = np.zeros(s.size, dtype=bool)
    bif_mask[lo:hi] = True

    fit_keep = ~bif_mask
    if int(fit_keep.sum()) < 4:
        raise ValueError(
            f"Only {int(fit_keep.sum())} points remain outside the bifurcation "
            f"window — too few to fit a PCHIP curve. Reduce n_bif_mask_nodes."
        )

    s_fit = s[fit_keep]
    r_fit = r[fit_keep]
    order = np.argsort(s_fit)
    interpolator = PchipInterpolator(s_fit[order], r_fit[order], extrapolate=True)
    r_interp = interpolator(s)
    diff = r - r_interp

    bulb_mask = _classify_bulb(diff, bulb_threshold_mm, bulb_expand_nodes)

    add_point_array(out, keep.astype(np.int32), "KeepAfterBlankingTrim")
    add_point_array(out, _expand_to_full(r_interp, keep, np.nan), "Radius MIS interp")
    add_point_array(out, _expand_to_full(diff, keep, np.nan), "Radius MIS diff (raw - interp)")

    bif_full = np.zeros(n_full, dtype=np.int32)
    bif_full[keep] = bif_mask.astype(np.int32)
    add_point_array(out, bif_full, "BifurcationMask")

    bulb_full = np.zeros(n_full, dtype=np.int32)
    bulb_full[keep] = bulb_mask.astype(np.int32)
    add_point_array(out, bulb_full, "BulbMask")

    return out


def _leading_trailing_blanking_keep_mask(blanking):
    """
    Returns a boolean mask (True = keep) that drops only the leading and
    trailing runs of ``blanking == 1`` from a centerline. Middle blanking
    points (e.g. the carotid bifurcation neighbourhood) are preserved.

    Parameters
    ----------
    blanking : np.ndarray
        1-D integer array of 0/1 blanking flags along the centerline.

    Returns
    -------
    keep : np.ndarray
        Boolean array of the same length, ``True`` for points to keep.

    """
    n = blanking.size
    keep = np.ones(n, dtype=bool)
    i = 0
    while i < n and blanking[i] == 1:
        keep[i] = False
        i += 1
    j = n - 1
    while j >= 0 and blanking[j] == 1:
        keep[j] = False
        j -= 1
    return keep


def _find_cca_ica_transition(vessel_type_name):
    """
    Returns the index of the first centerline point whose ``VesselTypeName``
    contains the substring ``"ICA"`` (matches both ``LICA`` and ``RICA``).

    Parameters
    ----------
    vessel_type_name : sequence of str
        Per-point vessel-type names along the centerline.

    Returns
    -------
    transition_idx : int
        Index of the first ICA-labelled point.

    Raises
    ------
    RuntimeError
        If no ICA-labelled point exists, the centerline is entirely ICA, or
        the centerline starts on ICA (no proximal CCA points to interpolate
        from).

    """
    is_ica = np.array(["ICA" in (n or "") for n in vessel_type_name])
    if not is_ica.any():
        raise RuntimeError("No ICA-labelled points on this centerline")
    if is_ica.all():
        raise RuntimeError(
            "Entire centerline is labelled ICA — no CCA→ICA transition"
        )
    transition_idx = int(np.argmax(is_ica))
    if transition_idx == 0:
        raise RuntimeError(
            "Centerline starts on ICA — nothing proximal to interpolate from"
        )
    return transition_idx


def _classify_bulb(diff, threshold_mm, expand_nodes_each_side):
    """
    Returns the bulb classification mask: ``|diff| > threshold_mm`` followed
    by symmetric run expansion of ``expand_nodes_each_side`` nodes per side.

    Parameters
    ----------
    diff : np.ndarray
        Residual ``raw_radius - r_interp`` per centerline node.
    threshold_mm : float
        Bulb-classification threshold (mm).
    expand_nodes_each_side : int
        Number of nodes by which each contiguous run of threshold-positive
        nodes is extended on each side. ``0`` disables expansion.

    Returns
    -------
    mask : np.ndarray
        Boolean array of the same length as ``diff``.

    """
    mask = np.abs(diff) > float(threshold_mm)
    if expand_nodes_each_side <= 0 or not mask.any():
        return mask
    expanded = mask.copy()
    for _ in range(int(expand_nodes_each_side)):
        left = np.concatenate(([False], expanded[:-1]))
        right = np.concatenate((expanded[1:], [False]))
        expanded = expanded | left | right
    return expanded


def _expand_to_full(trimmed, keep_mask, fill):
    """
    Expands a trimmed array (length ``keep_mask.sum()``) back to the original
    centerline length, filling ``fill`` at trimmed-out positions.

    Parameters
    ----------
    trimmed : np.ndarray
        1-D array of the trimmed signal.
    keep_mask : np.ndarray
        Boolean mask (length = original N) marking which positions were kept.
    fill : float
        Value to write at trimmed-out positions.

    Returns
    -------
    full : np.ndarray
        Float64 array of length ``keep_mask.size``.

    """
    full = np.full(keep_mask.size, fill, dtype=np.float64)
    full[keep_mask] = trimmed
    return full
