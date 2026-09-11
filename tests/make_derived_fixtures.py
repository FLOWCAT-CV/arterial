#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Regenerates the derived test fixtures from the input fixtures by running the
pipeline stages on them. The products land in
``tests/test_data/input_test_data/derived/extracranial_vessels`` (gitignored)
and are used by tests that need real execution products: a surface mesh,
several branch models, a featurised local graph, single-segment graphs,
landmarks and individual centerlines.

Usage, from the repository root with the environment active, ``arterial_dir``
set and the weights installed (a GPU makes it a couple of minutes):

    python tests/make_derived_fixtures.py [--force]
"""

import argparse
import os
import shutil
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from helpers import FIXTURES_DIR, fixture  # noqa: E402

MODE = "extracranial_vessels"
DERIVED_DIR = os.path.join(FIXTURES_DIR, "derived")


def main(force=False):
    from arterial.io.load_and_save_operations import load_json
    from arterial.centerline_extraction.centerline_extractor import CenterlineExtractor
    from arterial.vessel_labelling.vessel_labeller import VesselLabeller
    from arterial.feature_extraction.feature_extractor import FeatureExtractor
    from arterial.landmark_detection.landmark_detector import LandmarkDetector

    mode_dir = os.path.join(DERIVED_DIR, MODE)
    if os.path.isdir(mode_dir) and not force:
        print(f"{mode_dir} exists; pass --force to regenerate.")
        return 0
    shutil.rmtree(DERIVED_DIR, ignore_errors=True)
    os.makedirs(mode_dir)
    cta = fixture("cta.nii.gz")
    segmentation = fixture("segmentation.nii.gz")

    print("== centerlines (fast mode) ==")
    extractor = CenterlineExtractor(DERIVED_DIR, MODE, segmentation, fast_segmentation=True)
    extractor.perform_preprocessing()
    extractor.perform_centerline_extraction()
    extractor.perform_branch_model_extraction()
    extractor.perform_centerline_postprocessing()

    print("== vessel labelling ==")
    labeller = VesselLabeller(DERIVED_DIR, MODE, extractor.centerline_segments_array_path)
    labeller.build_segments_graph(save=True)
    labeller.predict_vessel_types(ensemble=True, save=True)

    print("== feature extraction ==")
    features = FeatureExtractor(DERIVED_DIR, MODE, 2, cta, extractor.centerline_segments_array_path,
                                extractor.branch_model_path, labeller.segments_graph_pred_path)
    features.build_local_graph()
    features.extract_local_features()
    features.extract_segment_features()
    features.extract_global_features()
    features.extract_supersegments()

    print("== landmarks ==")
    detector = LandmarkDetector(DERIVED_DIR, MODE, cta, segmentation_nifti_path=segmentation)
    detector.detect_landmarks_on_cta(return_mask=False, save=True)

    print("== individual centerlines ==")
    landmarks = load_json(detector.landmarks_ras_json_path)
    for centerline_id, (start, end) in {"l-ica": ("l-eica", "l-tica"), "r-ica": ("r-eica", "r-tica")}.items():
        try:
            extractor.extract_centerline_between_endpoints(landmarks[start], landmarks[end], centerline_id=centerline_id)
        except Exception as error:  # a failed individual centerline is not fatal for the fixture set
            print(f"WARNING: could not extract {centerline_id}: {error}")
    print(f"derived fixtures written to {mode_dir}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--force", action="store_true", help="regenerate even if the derived directory exists")
    sys.exit(main(force=parser.parse_args().force))
