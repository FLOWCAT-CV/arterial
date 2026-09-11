#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

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
    radius_array_name="Radius CE",
    bulb_landmark_world_mm=None,
    proximal_vessel_substring="CCA",
    bif_mask_width_mm=25.0,
    bulb_threshold_mm=(0.5, 0.2),
    bulb_expand_nodes=1,
):
    """
    Detects the carotid bulb on a CCA→ICA centerline polydata and adds the
    intermediate signals as point-data arrays.

    The algorithm proceeds in four steps:

    1. **Trim to the main proximal segment.** Drops everything before the
       start of the *longest* run of points satisfying both
       ``Blanking == 0`` and ``proximal_vessel_substring in VesselTypeName``
       (i.e. the longest contiguous proximal-vessel chunk). This handles
       short non-blanking stubs at the very start of the centerline that
       would otherwise become the trim boundary under a naive
       "first-non-blanking" rule. The trailing end drops the trailing run
       of ``Blanking == 1``. Middle blanking (the carotid bifurcation
       neighbourhood) is preserved.
    2. **Locate the bifurcation centre.** If ``bulb_landmark_world_mm`` is
       provided (e.g. the side's ``l-eica`` / ``r-eica`` landmark from the
       landmark detector), the centre is the centerline node geometrically
       closest to that 3-D point. Otherwise it falls back to the first node
       whose ``VesselTypeName`` contains ``"ICA"`` (matches both ``LICA`` and
       ``RICA``).
    3. **Interpolate the radius across the bifurcation.** Masks all nodes
       within an asymmetric arclength window around the centre: ``2/5``
       of ``bif_mask_width_mm`` proximal of the centre and ``3/5`` distal
       (so the 25 mm default spans 10 mm proximal + 15 mm distal toward
       the ICA — the bulb sits distal to the transition, so the ICA side
       gets the longer arm). A PCHIP curve is then fit through the
       surviving non-masked points and evaluated at every centerline
       arclength.
    4. **Classify bulb nodes.** Threshold-positive where
       ``|raw_radius - r_interp|`` exceeds a side-dependent threshold:
       proximal of the bifurcation centre, the proximal value of
       ``bulb_threshold_mm`` (or the scalar, if uniform); distal of the
       centre, the distal value. The default ``(0.4, 0.2)`` is stricter on
       the CCA / BT side (rejects mild radius wiggles where the bulb
       cannot be) and more permissive on the ICA side (catches the bulb's
       gentle distal taper). The bulb is then defined as the full span
       between the first and last threshold-positive node (interior dips
       below threshold are kept as bulb — the bulb is treated as a single
       connected segment). The span is then symmetrically extended by
       ``bulb_expand_nodes`` on each side via a 1-D binary dilation
       (clamped to the array boundaries).

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
        radius array named by ``radius_array_name``.
    radius_array_name : str, optional
        Name of the radius array on the centerline driving bulb detection.
        The default is ``"Radius CE"`` (cross-section-equivalent radius
        from ``perform_radius_extraction``; requires the case's surface
        mesh upstream). Pass ``"MaximumInscribedSphereRadius"`` to fall
        back to the VMTK MIS radius — useful when no surface mesh is
        available (e.g. centerlines built without segmentation.vtk).
    bulb_landmark_world_mm : sequence of float, optional
        World-coordinate ``(x, y, z)`` of the side's bifurcation landmark
        (typically ``l-eica`` for LCA and ``r-eica`` for RCA / BT-RCA from
        the landmark detector). When given, the bifurcation-mask centre is
        the centerline node closest to this point. When None, the centre is
        recovered from ``VesselTypeName`` via string matching on ``"ICA"``.
    proximal_vessel_substring : str, optional
        Substring used to identify the proximal vessel segment in
        ``VesselTypeName`` for the leading-edge trim. The default is
        ``"CCA"`` (matches ``LCCA`` and ``RCCA``); use ``"BT"`` for the
        BT-RCA centerline so the trim anchors on the brachiocephalic-trunk
        proximal end.
    bif_mask_width_mm : float, optional
        Total arclength width (mm) of the bifurcation mask, split
        asymmetrically as 2/5 proximal + 3/5 distal of the bifurcation
        centre (favouring the ICA side where the bulb sits). The default
        is 25.0 (10 mm proximal + 15 mm distal).
    bulb_threshold_mm : float or (float, float), optional
        Threshold for ``|raw - interp|``. As a scalar, applies uniformly
        across the centerline. As a 2-tuple ``(proximal, distal)``, the
        first value is used for nodes proximal of the bifurcation centre
        (CCA / BT side) and the second for nodes distal of it (ICA side,
        where the bulb sits). The bulb itself is the closed interval
        between the first and last threshold-positive node. The default
        is ``(0.4, 0.2)``.
    bulb_expand_nodes : int, optional
        Symmetric expansion of the bulb span, in nodes per side. The
        default is 1.

    Returns
    -------
    out : vtk.vtkPolyData
        Deep copy of ``centerline`` with the following point-data arrays added:

        - ``KeepAfterBlankingTrim`` (int 0/1): which original nodes survived
          the leading/trailing blanking trim.
        - ``Radius interp`` (float, 0 where trimmed): the PCHIP baseline
          radius across the bifurcation-mask window.
        - ``Radius diff (raw - interp)`` (float, 0 where trimmed): the
          residual ``raw - interp``.
        - ``BifurcationMask`` (int 0/1, 0 where trimmed): the
          arclength window around the bifurcation centre.
        - ``BulbMask`` (int 0/1, 0 where trimmed): the final bulb
          classification.

    Raises
    ------
    RuntimeError
        If the bifurcation centre cannot be located: in landmark mode, when
        the centerline has no kept nodes; in string-match mode, when no
        ICA-labelled points exist, the centerline is entirely ICA, or it
        starts on ICA (no CCA proximal to interpolate from).
    ValueError
        If too few centerline nodes survive the bifurcation mask to fit a
        PCHIP curve through.

    """
    out = vtk.vtkPolyData()
    out.DeepCopy(centerline)

    n_full = out.GetNumberOfPoints()

    blanking_full = vtk_to_numpy(out.GetPointData().GetArray("Blanking")).astype(int)
    s_full = vtk_to_numpy(out.GetPointData().GetArray("Distance from origin")).astype(float)
    r_full = vtk_to_numpy(out.GetPointData().GetArray(radius_array_name)).astype(float)
    points_full = np.array([out.GetPoint(i) for i in range(n_full)], dtype=float)

    vt_name_array = out.GetPointData().GetAbstractArray("VesselTypeName")
    if vt_name_array is None:
        raise RuntimeError(
            "VesselTypeName point-data array is required for carotid "
            "analysis but was not found on the input centerline."
        )
    vt_name_full = [vt_name_array.GetValue(i) for i in range(n_full)]

    keep = _trim_to_main_segment_keep_mask(
        vt_name_full, blanking_full, proximal_vessel_substring
    )
    s = s_full[keep]
    r = r_full[keep]
    points = points_full[keep]
    vt_name = [vt_name_full[i] for i in np.where(keep)[0]]

    if bulb_landmark_world_mm is not None:
        center_idx = _find_closest_centerline_idx(points, bulb_landmark_world_mm)
    else:
        center_idx = _find_cca_ica_transition(vt_name)

    # Asymmetric window: 2/5 of the total width proximal of the bifurcation
    # centre, 3/5 distal (toward the ICA). The bulb sits distal to the
    # CCA→ICA transition, so giving the ICA side more room keeps it inside
    # the PCHIP-hide zone.
    proximal_extent = float(bif_mask_width_mm) * 1.0 / 4.0
    distal_extent = float(bif_mask_width_mm) * 3.0 / 4.0
    delta_s = s - s[center_idx]
    bif_mask = (delta_s >= -proximal_extent) & (delta_s <= distal_extent)

    fit_keep = ~bif_mask
    if int(fit_keep.sum()) < 4:
        raise ValueError(
            f"Only {int(fit_keep.sum())} points remain outside the bifurcation "
            f"window — too few to fit a PCHIP curve. Reduce bif_mask_width_mm."
        )

    s_fit = s[fit_keep]
    r_fit = r[fit_keep]
    order = np.argsort(s_fit)
    interpolator = PchipInterpolator(s_fit[order], r_fit[order], extrapolate=True)
    r_interp = interpolator(s)
    diff = r - r_interp

    threshold_per_node = _build_per_node_threshold(bulb_threshold_mm, s, s[center_idx])
    bulb_mask = _classify_bulb(diff, threshold_per_node, bulb_expand_nodes)

    add_point_array(out, keep.astype(np.int32), "KeepAfterBlankingTrim")
    # Names are intentionally source-agnostic ("Radius interp" rather than
    # "Radius MIS interp") so they don't lie when `radius_array_name` is set
    # to e.g. "Radius CE".
    add_point_array(out, _expand_to_full(r_interp, keep, 0.0), "Radius interp")
    add_point_array(out, _expand_to_full(diff, keep, 0.0), "Radius diff (raw - interp)")

    bif_full = np.zeros(n_full, dtype=np.int32)
    bif_full[keep] = bif_mask.astype(np.int32)
    add_point_array(out, bif_full, "BifurcationMask")

    bulb_full = np.zeros(n_full, dtype=np.int32)
    bulb_full[keep] = bulb_mask.astype(np.int32)
    add_point_array(out, bulb_full, "BulbMask")

    return out


def _trim_to_main_segment_keep_mask(vessel_type_name, blanking, proximal_substring):
    """
    Returns a boolean keep mask that anchors the leading edge on the *longest*
    contiguous run of points satisfying both ``Blanking == 0`` and
    ``proximal_substring in VesselTypeName``, and drops the trailing run of
    ``Blanking == 1``. Middle blanking (e.g. the carotid bifurcation
    neighbourhood) is preserved.

    The longest-run rule for the proximal end is robust to short non-blanking
    stubs at the very start of the centerline — those would otherwise become
    the trim boundary under a naive "first non-blanking point" rule and bias
    the PCHIP baseline.

    Parameters
    ----------
    vessel_type_name : sequence of str
        Per-point ``VesselTypeName`` along the centerline.
    blanking : np.ndarray
        1-D integer array of 0/1 blanking flags along the centerline.
    proximal_substring : str
        Substring used to identify the proximal vessel segment (e.g.
        ``"CCA"`` for LCA / RCA, ``"BT"`` for BT-RCA).

    Returns
    -------
    keep : np.ndarray
        Boolean array of the same length, ``True`` for points to keep.

    Raises
    ------
    RuntimeError
        If no point matches both criteria (no proximal-vessel anchor found).

    """
    n = blanking.size

    is_proximal = np.array(
        [
            proximal_substring in (name or "") and bl == 0
            for name, bl in zip(vessel_type_name, blanking)
        ],
        dtype=bool,
    )
    if not is_proximal.any():
        raise RuntimeError(
            f"No non-blanking points with VesselTypeName containing "
            f"{proximal_substring!r} were found on the centerline; cannot "
            f"locate the proximal trim anchor."
        )

    edges = np.diff(np.concatenate(([0], is_proximal.astype(np.int8), [0])))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0]  # exclusive
    longest = int(np.argmax(ends - starts))
    start_idx = int(starts[longest])

    j = n - 1
    while j >= 0 and blanking[j] == 1:
        j -= 1
    end_idx = j + 1  # exclusive

    keep = np.zeros(n, dtype=bool)
    keep[start_idx:end_idx] = True
    return keep


def _find_closest_centerline_idx(points, world_mm):
    """
    Returns the index of the centerline point closest to ``world_mm`` in 3-D
    Euclidean distance.

    Parameters
    ----------
    points : np.ndarray
        ``(N, 3)`` float array of centerline-point world coordinates.
    world_mm : sequence of float
        World-coordinate ``(x, y, z)`` to query.

    Returns
    -------
    idx : int
        Index of the closest centerline point.

    Raises
    ------
    RuntimeError
        If ``points`` is empty.

    """
    if points.size == 0:
        raise RuntimeError(
            "No centerline nodes available to match the bulb landmark against."
        )
    target = np.asarray(world_mm, dtype=float).reshape(3)
    return int(np.argmin(np.linalg.norm(points - target, axis=1)))


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


def _build_per_node_threshold(bulb_threshold_mm, s, s_center):
    """
    Resolves the per-node bulb-classification threshold.

    A scalar yields a uniform threshold; a 2-tuple ``(proximal, distal)``
    yields the proximal value for nodes with arclength below ``s_center``
    and the distal value otherwise.

    Parameters
    ----------
    bulb_threshold_mm : float or sequence of two floats
        Scalar (uniform) or ``(proximal, distal)`` pair.
    s : np.ndarray
        Per-node arclength along the centerline (mm).
    s_center : float
        Arclength of the bifurcation centre (mm).

    Returns
    -------
    threshold : float or np.ndarray
        Either a Python float (scalar case) or a per-node float array.

    """
    if np.ndim(bulb_threshold_mm) == 0:
        return float(bulb_threshold_mm)
    if len(bulb_threshold_mm) != 2:
        raise ValueError(
            f"bulb_threshold_mm must be a scalar or a 2-tuple "
            f"(proximal, distal); got {bulb_threshold_mm!r}."
        )
    thr_proximal = float(bulb_threshold_mm[0])
    thr_distal = float(bulb_threshold_mm[1])
    return np.where(s < s_center, thr_proximal, thr_distal)


def _classify_bulb(diff, threshold_mm, expand_nodes_each_side):
    """
    Returns the bulb classification mask. Threshold-positive nodes
    (``|diff| > threshold_mm``) are first bridged into a single span — every
    node between the first and last positive is labelled bulb, regardless of
    whether it itself crosses the threshold — then the span is symmetrically
    extended by ``expand_nodes_each_side`` nodes on each side.

    Bridging makes detection robust to localised dips inside the bulb (e.g.
    plaque or smoothing artefacts that briefly bring ``|diff|`` back under
    threshold mid-bulb) at the cost of treating the bulb as a single connected
    segment — so any spurious threshold crossing far from the true bulb will
    drag the detected span to it.

    Parameters
    ----------
    diff : np.ndarray
        Residual ``raw_radius - r_interp`` per centerline node.
    threshold_mm : float or np.ndarray
        Bulb-classification threshold (mm). Either a scalar applied
        uniformly, or a per-node array (e.g. for a side-dependent
        threshold). Compared via numpy broadcasting against ``|diff|``.
    expand_nodes_each_side : int
        Number of nodes by which the bridged bulb span is extended on each
        side. ``0`` disables expansion.

    Returns
    -------
    mask : np.ndarray
        Boolean array of the same length as ``diff``.

    """
    threshold_mask = np.abs(diff) > np.asarray(threshold_mm, dtype=float)
    mask = np.zeros_like(threshold_mask)
    if threshold_mask.any():
        first = int(np.argmax(threshold_mask))
        last = int(threshold_mask.size - 1 - np.argmax(threshold_mask[::-1]))
        mask[first : last + 1] = True
    if expand_nodes_each_side <= 0 or not mask.any():
        return mask
    expanded = mask.copy()
    for _ in range(int(expand_nodes_each_side)):
        left = np.concatenate(([False], expanded[:-1]))
        right = np.concatenate((expanded[1:], [False]))
        # expanded = expanded | left | right
        expanded = expanded | right
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
