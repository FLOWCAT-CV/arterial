#!/usr/bin/env python3
#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0
"""
Run the full arterial pipeline on a case directory and export per-vessel VTK
centerlines with full geometry arrays (MIS / CE / CC radii, ovality, curvature,
filtered curvature, torsion, distance from origin, angle of curvature, cumulative
angle of curvature, blanking, Frenet-Serret frame), plus — on the carotid
centerlines — the carotid bulb analysis arrays.

For each of RCCA, RCA, RICA, LCCA, LCA, LICA, BT-RCA whose
`single_segments/{vessel}.pickle` is produced by feature extraction, the script:

  1. Converts the pickle to a vtkPolyData centerline in native NIfTI/VTK space
     (`FeatureExtractor.single_segment_pickle_to_vtk`).
  2. Augments it with cross-section radii (using the case's `segmentation.vtk`
     surface mesh) and curvature arrays (`FeatureExtractor.add_centerline_geometry`).
  3. **For LCA / RCA / BT-RCA only**, runs the carotid bulb analysis
     (`FeatureExtractor.add_carotid_analysis`), which appends
     `KeepAfterBlankingTrim`, `Radius interp`, `Radius diff (raw - interp)`,
     `BifurcationMask`, and `BulbMask`. With `--detect-intracranial` the same
     call also appends `Intracranial` and `DistanceTransformValueSmoothed`
     (the cranium-DT cache makes this a one-time cost per case).
  4. Saves to `{case_dir}/{mode}/individual_centerlines/{vessel}.vtk`.

A failure on one vessel is logged but does not abort processing of the others;
an end-of-run summary string reports per-vessel outcomes.

The full pipeline is invoked through `ArterialProcessor` exactly as in
the `arterial` command; the geometry export is a post-processing step layered on top
and does not modify the processor.
"""
from __future__ import annotations

import argparse
import os
import traceback
from argparse import Namespace

from arterial.run.processor import ArterialProcessor
from arterial.feature_extraction.feature_extractor import FeatureExtractor
from arterial.io.load_and_save_operations import load_pickle, load_vtkpolydata


TARGET_VESSELS = ["RCCA", "RCA", "RICA", "LCCA", "LCA", "LICA", "BT-RCA"]
# Vessels that span CCA → ICA — `add_carotid_analysis` only makes sense on
# these. BT-RCA is the brachiocephalic-trunk → right-carotid path.
CAROTID_BULB_VESSELS = {"LCA", "RCA", "BT-RCA"}


def export_vessel_geometry(
    case_dir: str,
    mode: str,
    sampling_distance_mm: float,
    cta_nifti_path: str | None,
    detect_intracranial: bool = False,
) -> list[dict]:
    """For each available per-vessel pickle, build the geometry-augmented
    centerline VTK and (for the carotid subset) append the bulb arrays.

    Returns a per-vessel results list with one dict per entry in
    ``TARGET_VESSELS`` carrying ``geometry`` ∈ {"ok", "missing", "failed"}
    and ``bulb`` ∈ {"ok", "failed", None} (None when the bulb step does not
    apply or when the geometry step itself failed).
    """
    fe = FeatureExtractor(case_dir=case_dir, mode=mode, sampling_distance_mm=sampling_distance_mm, cta_nifti_path=cta_nifti_path)

    surface_path = os.path.join(case_dir, mode, "segmentation.vtk")
    if os.path.isfile(surface_path):
        surface = load_vtkpolydata(surface_path)
    else:
        surface = None
        print(f"WARNING: surface mesh not found at {surface_path}; cross-section radii (CE, CC, ovality) will be skipped.")

    out_dir = os.path.join(case_dir, mode, "individual_centerlines")
    os.makedirs(out_dir, exist_ok=True)

    single_segments_dir = os.path.join(case_dir, mode, "single_segments")
    results: list[dict] = []
    for vessel in TARGET_VESSELS:
        result: dict = {"vessel": vessel, "geometry": None, "bulb": None}
        pickle_path = os.path.join(single_segments_dir, f"{vessel}.pickle")
        if not os.path.isfile(pickle_path):
            print(f"  {vessel}: pickle not found at {pickle_path}, skipping")
            result["geometry"] = "missing"
            results.append(result)
            continue
        # Wrap the whole per-vessel block: a failure on one vessel (e.g.
        # BT-RCA) must not abort processing of the remaining vessels for
        # the same case.
        try:
            graph = load_pickle(pickle_path)
            centerline = fe.single_segment_pickle_to_vtk(graph)
            out_path = os.path.join(out_dir, f"{vessel}.vtk")
            out = fe.add_centerline_geometry(centerline, surface_model=surface, save_path=out_path)
            n_pts = out.GetNumberOfPoints()
            result["geometry"] = "ok"

            extras: list[str] = []
            if vessel in CAROTID_BULB_VESSELS:
                try:
                    # `add_carotid_analysis` re-saves the same file with the
                    # bulb arrays appended. `side=vessel` triggers the
                    # side-aware landmark resolution (l-eica / r-eica) and
                    # proximal-substring trim ("CCA" for LCA/RCA, "BT" for
                    # BT-RCA). `detect_intracranial` here lets the cranium-DT
                    # cache pay its cost only once per case.
                    fe.add_carotid_analysis(
                        out,
                        side=vessel,
                        detect_intracranial=detect_intracranial,
                        save_path=out_path,
                    )
                    extras.append("bulb")
                    if detect_intracranial:
                        extras.append("intracranial")
                    result["bulb"] = "ok"
                except Exception as exc:  # noqa: BLE001
                    # Finer-grained skip: keep the geometry export even if
                    # the bulb step fails (e.g. degenerate vessel labelling
                    # produced no valid CCA→ICA span).
                    print(f"  {vessel}: carotid analysis skipped ({exc})")
                    result["bulb"] = "failed"

            suffix = (" +" + "+".join(extras)) if extras else ""
            print(f"  {vessel}: wrote {out_path} ({n_pts} points){suffix}")
        except Exception as exc:  # noqa: BLE001
            print(f"  {vessel}: FAILED, continuing with next vessel ({exc})")
            traceback.print_exc()
            result["geometry"] = "failed"

        results.append(result)
    return results


def summarize_results(results: list[dict]) -> str:
    """Compress per-vessel outcomes into a one-line case status string, e.g.
    ``"7/7 vessels | bulb_failed: BT-RCA"`` or
    ``"5/7 vessels | missing: LCCA, LICA | geom_failed: BT-RCA"``.
    """
    n_total = len(TARGET_VESSELS)
    n_geom_ok = sum(1 for r in results if r["geometry"] == "ok")
    missing = [r["vessel"] for r in results if r["geometry"] == "missing"]
    geom_failed = [r["vessel"] for r in results if r["geometry"] == "failed"]
    bulb_failed = [r["vessel"] for r in results if r["bulb"] == "failed"]

    parts = [f"{n_geom_ok}/{n_total} vessels"]
    if missing:
        parts.append("missing: " + ", ".join(missing))
    if geom_failed:
        parts.append("geom_failed: " + ", ".join(geom_failed))
    if bulb_failed:
        parts.append("bulb_failed: " + ", ".join(bulb_failed))
    return " | ".join(parts)


def build_processor_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        case_dir=args.case_dir,
        cta_nifti_path=args.cta_nifti_path,
        mode=args.mode,
        sampling_distance_mm=args.sampling_distance_mm,
        fast_segmentation=args.fast_segmentation,
        skip_segmentation=args.skip_segmentation,
        skip_centerline_extraction=args.skip_centerline_extraction,
        skip_branching=args.skip_branching,
        skip_clipping=args.skip_clipping,
        skip_vessel_labelling=args.skip_vessel_labelling,
        skip_feature_extraction=args.skip_feature_extraction,
        skip_access_prediction=args.skip_access_prediction,
        skip_landmark_detection=args.skip_landmark_detection,
        cl_dice_nnunet=args.cl_dice_nnunet,
        no_slicing=args.no_slicing,
        set_threshold_099=args.threshold_099,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-cd", "--case_dir", type=str, required=True,
        help="Path to case directory.")
    parser.add_argument("-cnp", "--cta_nifti_path", type=str, default=None,
        help="Path to the CTA NIfTI. Defaults to {case_dir}/cta.nii.gz.")
    parser.add_argument("-m", "--mode", type=str, default="extracranial_vessels",
        choices=["extracranial_vessels", "intracranial_vessels"],
        help="Pipeline mode. Default: extracranial_vessels.")
    parser.add_argument("-sd", "--sampling_distance_mm", type=float, default=2,
        help="Centerline sampling distance in mm. Default: 2.")

    seg_group = parser.add_argument_group("segmentation source")
    seg_mx = seg_group.add_mutually_exclusive_group()
    seg_mx.add_argument("--default-segmentation", dest="default_segmentation", action="store_true",
        help="Use the default nnU-Net segmentation (standard threshold). This is the default.")
    seg_mx.add_argument("--threshold-099", dest="threshold_099", action="store_true",
        help="Use the segmentation derived from the probability map at the 0.99 threshold "
             "(more conservative; retains only voxels the model is highly confident about).")

    pipeline_group = parser.add_argument_group("pipeline toggles (forwarded to ArterialProcessor)")
    pipeline_group.add_argument("-fast", "--fast_segmentation", action="store_true",
        help="Use fast (single-resolution) segmentation.")
    pipeline_group.add_argument("-ss", "--skip_segmentation", action="store_true",
        help="Skip segmentation (requires an existing {mode}_segmentation.nii.gz in case_dir).")
    pipeline_group.add_argument("-sce", "--skip_centerline_extraction", action="store_true")
    pipeline_group.add_argument("-sb", "--skip_branching", action="store_true")
    pipeline_group.add_argument("-sc", "--skip_clipping", action="store_true")
    pipeline_group.add_argument("-svl", "--skip_vessel_labelling", action="store_true")
    pipeline_group.add_argument("-sfe", "--skip_feature_extraction", action="store_true")
    pipeline_group.add_argument("-sap", "--skip_access_prediction", action="store_true")
    pipeline_group.add_argument("-sld", "--skip_landmark_detection", action="store_true")
    pipeline_group.add_argument("-clnn", "--cl_dice_nnunet", action="store_true",
        help="Use the centerline-Dice-trained nnU-Net variant.")
    pipeline_group.add_argument("-ns", "--no_slicing", action="store_true",
        help="Disable image slicing (intracranial-vessel head CTAs).")

    parser.add_argument("--skip-pipeline", action="store_true",
        help="Skip the full ArterialProcessor pipeline and only run the per-vessel VTK export. "
             "Use this when single_segments/ pickles already exist from a prior run.")
    parser.add_argument("--detect-intracranial", action="store_true",
        help="On the carotid centerlines (LCA, RCA, BT-RCA), additionally run the "
             "intracranial-transition detector inside add_carotid_analysis and append "
             "the Intracranial and DistanceTransformValueSmoothed point-data arrays. "
             "Off by default because the underlying cranium distance transform "
             "is expensive (~30 s on a typical head-and-neck CTA); the result "
             "is cached on disk under {case_dir}/{mode}/cranium_distance_transform.npy "
             "so subsequent runs reuse it.")

    args = parser.parse_args()

    seg_mode = "0.99 probability-map threshold" if args.threshold_099 else "default"
    print(f"=== Arterial pipeline + per-vessel geometry export + bulb ===")
    print(f"    case_dir   : {args.case_dir}")
    print(f"    mode       : {args.mode}")
    print(f"    segmentation: {seg_mode}")

    if not args.skip_pipeline:
        processor_args = build_processor_args(args)
        processor = ArterialProcessor(processor_args)
        processor.perform_analysis()
    else:
        print("Skipping ArterialProcessor (--skip-pipeline); using existing single_segments pickles.")

    print(f"\n=== Exporting per-vessel VTK centerlines ===")
    results = export_vessel_geometry(
        args.case_dir,
        args.mode,
        args.sampling_distance_mm,
        args.cta_nifti_path,
        detect_intracranial=args.detect_intracranial,
    )
    print(f"\n=== Summary: {summarize_results(results)} ===")


if __name__ == "__main__":
    main()
