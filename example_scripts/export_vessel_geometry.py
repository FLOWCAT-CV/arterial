#!/usr/bin/env python3
"""
Run the full arterial pipeline on a case directory and export per-vessel VTK
centerlines with full geometry arrays (MIS / CE / CC radii, ovality, curvature,
filtered curvature, torsion, distance from origin, angle of curvature, cumulative
angle of curvature, blanking, Frenet-Serret frame).

For each of RCCA, RCA, RICA, LCCA, LCA, LICA whose `single_segments/{vessel}.pickle`
is produced by feature extraction, the script:
  1. Converts the pickle to a vtkPolyData centerline in native NIfTI/VTK space
     (`FeatureExtractor.single_segment_pickle_to_vtk`).
  2. Augments it with cross-section radii (using the case's `segmentation.vtk`
     surface mesh) and curvature arrays (`FeatureExtractor.add_centerline_geometry`).
  3. Saves to `{case_dir}/{mode}/individual_centerlines/{vessel}.vtk`.

The full pipeline is invoked through `ArterialProcessor` exactly as in
`perform_analysis.py`; the geometry export is a post-processing step layered on top
and does not modify the processor.
"""
from __future__ import annotations

import argparse
import os
from argparse import Namespace

from arterial.run.processor import ArterialProcessor
from arterial.feature_extraction.feature_extractor import FeatureExtractor
from arterial.io.load_and_save_operations import load_pickle, load_vtkpolydata


TARGET_VESSELS = ["RCCA", "RCA", "RICA", "LCCA", "LCA", "LICA"]
CAROTID_VESSELS = {"LCA", "RCA"}


def export_vessel_geometry(
    case_dir: str,
    mode: str,
    sampling_distance_mm: float,
    cta_nifti_path: str | None,
    detect_intracranial: bool = False,
) -> None:
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
    for vessel in TARGET_VESSELS:
        pickle_path = os.path.join(single_segments_dir, f"{vessel}.pickle")
        if not os.path.isfile(pickle_path):
            print(f"  {vessel}: pickle not found at {pickle_path}, skipping")
            continue
        graph = load_pickle(pickle_path)
        centerline = fe.single_segment_pickle_to_vtk(graph)
        out_path = os.path.join(out_dir, f"{vessel}.vtk")
        out = fe.add_centerline_geometry(centerline, surface_model=surface, save_path=out_path)
        msg = f"  {vessel}: wrote {out_path} ({out.GetNumberOfPoints()} points)"
        if detect_intracranial and vessel in CAROTID_VESSELS:
            # Re-saves the same file with `Intracranial` and
            # `DistanceTransformValueSmoothed` arrays appended. The cranium
            # distance transform is computed once (LCA) and reused on RCA via
            # the FeatureExtractor's three-tier cache.
            fe.add_intracranial_transition(out, save_path=out_path)
            msg += " +intracranial"
        print(msg)


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
        help="On the carotid centerlines (LCA, RCA), additionally run the "
             "intracranial-transition detector and append the Intracranial and "
             "DistanceTransformValueSmoothed point-data arrays to their VTKs. "
             "Off by default because the underlying cranium distance transform "
             "is expensive (~30 s on a typical head-and-neck CTA); the result "
             "is cached on disk under {case_dir}/{mode}/cranium_distance_transform.npy "
             "so subsequent runs reuse it.")

    args = parser.parse_args()

    seg_mode = "0.99 probability-map threshold" if args.threshold_099 else "default"
    print(f"=== Arterial pipeline + per-vessel geometry export ===")
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
    export_vessel_geometry(
        args.case_dir,
        args.mode,
        args.sampling_distance_mm,
        args.cta_nifti_path,
        detect_intracranial=args.detect_intracranial,
    )
    print("Done.")


if __name__ == "__main__":
    main()
