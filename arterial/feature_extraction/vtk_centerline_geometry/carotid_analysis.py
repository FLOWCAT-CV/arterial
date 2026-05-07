#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import vtk
import numpy as np

from vtk.util.numpy_support import vtk_to_numpy
from scipy.interpolate import PchipInterpolator
from scipy.ndimage import distance_transform_edt, gaussian_filter

from arterial.feature_extraction.vtk_centerline_geometry.utils import add_point_array
from arterial.segmentation.utils import compute_cranium_mask


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
    ``0`` in the radius/diff arrays (and in all binary masks); use
    ``KeepAfterBlankingTrim`` as the mask in ParaView to ignore them. (We
    avoid ``NaN`` here because the legacy ASCII VTK writer round-trips
    ``NaN`` as the literal token ``nan``, which trips newer ParaView
    readers.)

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
        - ``Radius MIS interp`` (float, 0 where trimmed): the PCHIP baseline
          radius across the bifurcation-mask window.
        - ``Radius MIS diff (raw - interp)`` (float, 0 where trimmed): the
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
    add_point_array(out, _expand_to_full(r_interp, keep, 0.0), "Radius MIS interp")
    add_point_array(out, _expand_to_full(diff, keep, 0.0), "Radius MIS diff (raw - interp)")

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


def perform_intracranial_transition_detection(
    centerline,
    cta_array,
    cta_affine,
    *,
    distance_transform=None,
    smoothing_sigma=2,
    dt_max_mm=20,
    dt_tolerance_mm=5,
    dt_clip_mm=50,
):
    """
    Detects the intracranial transition on a CCA→ICA centerline polydata.

    Computes (or reuses) a clipped cranium distance transform, samples it at
    each centerline point, smooths along arclength with a Gaussian filter, and
    locates the first downstream local minimum (sign-change of the gradient
    paired with second-derivative > 0) where the smoothed DT value falls below
    ``dt_max_mm``. Among qualifying minima, the first one whose DT value is
    within ``dt_tolerance_mm`` of the absolute minimum is selected — this
    filters out spurious early minima at high DT values. Centerline nodes at
    and proximal to the selected node are labelled extracranial; distal nodes
    are intracranial. When no qualifying minimum is found, all nodes are
    labelled extracranial.

    Centerline points are assumed to be in **native world coordinates** (the
    convention produced by ``pickle_to_vtk``); voxel indices are obtained via
    ``round(inv(cta_affine) @ [x, y, z, 1])`` with no orientation branching.

    Parameters
    ----------
    centerline : vtk.vtkPolyData
        Per-vessel centerline polydata.
    cta_array : numpy.ndarray
        3D numpy array with the CTA image.
    cta_affine : numpy.ndarray
        4x4 affine matrix of the CTA.
    distance_transform : numpy.ndarray, optional
        Pre-computed clipped cranium distance transform on the full CTA grid
        (shape == ``cta_array.shape``). If ``None`` it is computed locally.
        Pass this in when running the detection multiple times on the same
        case (e.g. LCA + RCA) to avoid recomputing the EDT.
    smoothing_sigma : float, optional
        Sigma (in nodes) of the Gaussian smoothing applied to the per-point DT
        signal before differentiation. The default is 2.
    dt_max_mm : float, optional
        A local minimum is only considered intracranial-transition-eligible
        when its smoothed DT value is below this threshold. The default is 20.
    dt_tolerance_mm : float, optional
        Among qualifying minima, only those within this tolerance of the
        absolute minimum DT value are kept; the first such minimum is the
        transition. The default is 5.
    dt_clip_mm : float, optional
        Maximum value used to clip the EDT (and to fill the lower half of the
        CTA grid where the DT is not computed). The default is 50.

    Returns
    -------
    out : vtk.vtkPolyData
        Deep copy of ``centerline`` with the following point-data arrays added:

        - ``Intracranial`` (int 0/1): per-point intracranial flag.
        - ``DistanceTransformValueSmoothed`` (float, mm): the Gaussian-smoothed
          per-point DT signal that drives detection.

    transition_world : numpy.ndarray or None
        World-coordinate (x, y, z) of the transition centerline point, or
        ``None`` when no transition was found.
    transition_ijk : numpy.ndarray or None
        Voxel-coordinate (i, j, k) of the transition point, or ``None`` when
        no transition was found.
    distance_transform : numpy.ndarray
        The full-CTA-grid clipped distance transform actually used (the input
        if provided, else the freshly computed one). Cache this if you intend
        to call again on the same case.

    """
    out = vtk.vtkPolyData()
    out.DeepCopy(centerline)

    if distance_transform is None:
        distance_transform = _compute_cranium_distance_transform(cta_array, dt_clip_mm)

    dt_values = _sample_dt_along_centerline(out, cta_affine, distance_transform)
    dt_smoothed = gaussian_filter(dt_values, sigma=smoothing_sigma)

    transition_idx = _find_first_local_minimum_idx(dt_smoothed, dt_max_mm, dt_tolerance_mm)

    n_points = out.GetNumberOfPoints()
    intracranial = np.zeros(n_points, dtype=np.int32)
    if transition_idx is not None:
        intracranial[transition_idx + 1:] = 1

    add_point_array(out, intracranial, "Intracranial")
    add_point_array(out, dt_smoothed.astype(np.float64), "DistanceTransformValueSmoothed")

    transition_world = None
    transition_ijk = None
    if transition_idx is not None:
        transition_world = np.array(out.GetPoint(transition_idx))
        transition_ijk = np.round(
            np.linalg.inv(cta_affine) @ np.append(transition_world, 1)
        )[:3].astype(int)

    return out, transition_world, transition_ijk, distance_transform


def _compute_cranium_distance_transform(cta_array, dt_clip_mm):
    """
    Computes the cranium-distance-transform field used by intracranial-
    transition detection.

    Builds a binary cranium mask via :func:`compute_cranium_mask` (LoG +
    largest-connected-component on the upper half of the CTA), takes the
    Euclidean distance transform of its complement, embeds the result back
    into a full-CTA-shaped grid (lower half filled with ``dt_clip_mm``), and
    clips to ``[0, dt_clip_mm]``.

    Parameters
    ----------
    cta_array : numpy.ndarray
        3D numpy array with the CTA image.
    dt_clip_mm : float
        Distance-transform clipping value (mm).

    Returns
    -------
    distance_transform : numpy.ndarray
        Float array of shape ``cta_array.shape``.

    """
    half_s_coordinate = cta_array.shape[2] // 2
    cranium_mask = compute_cranium_mask(cta_array)
    upper_half_dt = distance_transform_edt(1 - cranium_mask)
    distance_transform = np.full_like(cta_array, np.max(upper_half_dt), dtype=np.float64)
    distance_transform[:, :, half_s_coordinate:] = upper_half_dt
    distance_transform = np.clip(distance_transform, 0, dt_clip_mm)
    return distance_transform


def _sample_dt_along_centerline(centerline, cta_affine, distance_transform):
    """
    Samples a 3D scalar field at every centerline point.

    Centerline points are assumed to be in native world coordinates; voxel
    indices are obtained via ``round(inv(cta_affine) @ [x, y, z, 1])``. Points
    that fall outside the volume are clamped to the nearest valid voxel.

    Parameters
    ----------
    centerline : vtk.vtkPolyData
        Per-vessel centerline polydata.
    cta_affine : numpy.ndarray
        4x4 affine matrix of the CTA.
    distance_transform : numpy.ndarray
        3D scalar field defined on the CTA voxel grid.

    Returns
    -------
    values : numpy.ndarray
        1-D float array of length ``centerline.GetNumberOfPoints()``.

    """
    n_points = centerline.GetNumberOfPoints()
    inv_affine = np.linalg.inv(cta_affine)
    shape = distance_transform.shape

    values = np.empty(n_points, dtype=np.float64)
    for idx in range(n_points):
        world = np.array(centerline.GetPoint(idx))
        ijk = np.round(inv_affine @ np.append(world, 1))[:3].astype(int)
        ijk[0] = np.clip(ijk[0], 0, shape[0] - 1)
        ijk[1] = np.clip(ijk[1], 0, shape[1] - 1)
        ijk[2] = np.clip(ijk[2], 0, shape[2] - 1)
        values[idx] = distance_transform[ijk[0], ijk[1], ijk[2]]
    return values


def _find_first_local_minimum_idx(dt_smoothed, dt_max_mm, dt_tolerance_mm):
    """
    Returns the index of the first downstream local minimum of a 1-D signal
    whose value is below ``dt_max_mm`` and within ``dt_tolerance_mm`` of the
    absolute minimum among qualifying minima. Returns ``None`` when no such
    minimum exists.

    A local minimum is detected by a sign-change of the gradient (from
    negative to non-negative) paired with second-derivative > 0. The selection
    rule (within-tolerance-of-absolute-minimum, then first) matches the
    sandbox implementation in ``bulb_extraction/compute_cer_processing.py``.

    Parameters
    ----------
    dt_smoothed : numpy.ndarray
        1-D smoothed DT signal along the centerline.
    dt_max_mm : float
        Upper bound on the DT value at a qualifying minimum (mm).
    dt_tolerance_mm : float
        Tolerance around the absolute minimum DT value (mm).

    Returns
    -------
    transition_idx : int or None
        Index of the selected local minimum, or ``None``.

    """
    derivative = np.gradient(dt_smoothed)
    second_derivative = np.gradient(derivative)

    diff_sign = np.diff(np.sign(derivative))
    if diff_sign.size == 0:
        return None
    local_minima_idx = np.where((diff_sign != 0) & (second_derivative[1:] > 0))[0]
    if local_minima_idx.size == 0:
        return None

    qualifying = local_minima_idx[dt_smoothed[local_minima_idx] < dt_max_mm]
    if qualifying.size == 0:
        # No minimum within the DT-max gate; fall back to the first detected
        # local minimum so the caller still gets a transition rather than
        # silently dropping the result.
        return int(local_minima_idx[0])

    min_dt_value = np.min(dt_smoothed[qualifying])
    near_absolute = qualifying[
        dt_smoothed[qualifying] < min_dt_value + dt_tolerance_mm
    ]
    return int(near_absolute[0])
