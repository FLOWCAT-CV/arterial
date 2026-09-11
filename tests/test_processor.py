#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import argparse
import json
import os
import re
import subprocess
import sys

from helpers import ArterialTestCase
from arterial.run.processor import ArterialProcessor, SUPPORTED_MODES

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "example_scripts"))
import perform_analysis  # noqa: E402
import arterial_processing_full_pipeline  # noqa: E402

SKIP_FLAGS = ["skip_segmentation", "skip_centerline_extraction", "skip_branching", "skip_clipping", "skip_vessel_labelling",
              "skip_feature_extraction", "skip_access_prediction", "skip_landmark_detection"]


def all_skip_namespace(case_dir, cta_nifti_path, mode="extracranial_vessels"):
    return argparse.Namespace(case_dir=case_dir, cta_nifti_path=cta_nifti_path, mode=mode, sampling_distance_mm=2.0,
                              fast_segmentation=True, cl_dice_nnunet=False, no_slicing=False, set_threshold_099=False,
                              **{flag: True for flag in SKIP_FLAGS})


class TestArterialProcessor(ArterialTestCase):
    """Wiring of the processor without running any stage."""

    def setUp(self):
        super().setUp()
        self.cta_nifti_path = self.require_fixture("cta.nii.gz")

    def test_init_wires_every_module_and_flag(self):
        processor = ArterialProcessor(all_skip_namespace(self.case_dir, self.cta_nifti_path))
        self.assertEqual(processor.vessel_segmenter.cta_nifti_path, self.cta_nifti_path)
        self.assertEqual(processor.centerline_extractor.mode, "extracranial_vessels")
        self.assertEqual(processor.feature_extractor.sampling_distance_mm, 2.0)
        self.assertEqual(processor.access_predictor.access, ["femoral"])
        self.assertTrue(processor.use_vanilla_nnunet)
        for flag in SKIP_FLAGS:
            self.assertTrue(getattr(processor, flag), flag)

    def test_cta_path_defaults_to_case_dir(self):
        processor = ArterialProcessor(argparse.Namespace(case_dir=self.case_dir))
        self.assertEqual(processor.cta_nifti_path, os.path.join(self.case_dir, "cta.nii.gz"))
        self.assertFalse(processor.skip_segmentation)
        self.assertEqual(processor.mode, "extracranial_vessels")

    def test_unsupported_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            ArterialProcessor(all_skip_namespace(self.case_dir, self.cta_nifti_path, mode="thrombus"))

    def test_perform_analysis_with_everything_skipped(self):
        processor = ArterialProcessor(all_skip_namespace(self.case_dir, self.cta_nifti_path))
        times = processor.perform_analysis()
        expected = {"segmentation_time", "centerline_extraction_time", "landmark_detection_time", "vessel_labelling_time",
                    "feature_extraction_time", "access_prediction_time", "total_time"}
        self.assertEqual(set(times), expected)
        for key, value in times.items():
            self.assertLess(value, 5.0, f"{key} should be near zero when skipped")
        self.assertFalse(os.path.exists(self.mode_dir) and os.listdir(self.mode_dir), "skipping everything must write nothing")


class TestCommandLine(ArterialTestCase):
    """perform_analysis.py and the README's option table."""

    def test_parser_defaults_match_processor_expectations(self):
        args = perform_analysis.build_parser().parse_args(["-cd", self.case_dir])
        for flag in SKIP_FLAGS + ["fast_segmentation", "cl_dice_nnunet", "no_slicing", "set_threshold_099"]:
            self.assertFalse(getattr(args, flag), flag)
        self.assertEqual(args.mode, "extracranial_vessels")
        self.assertEqual(args.sampling_distance_mm, 2)

    def test_parser_rejects_unknown_mode(self):
        with self.assertRaises(SystemExit):
            perform_analysis.build_parser().parse_args(["-cd", self.case_dir, "-m", "thrombus"])
        for mode in SUPPORTED_MODES:
            self.assertEqual(perform_analysis.build_parser().parse_args(["-cd", self.case_dir, "-m", mode]).mode, mode)

    def test_help_runs_as_a_script(self):
        result = subprocess.run([sys.executable, os.path.join(REPO_ROOT, "perform_analysis.py"), "--help"],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--skip_segmentation", result.stdout)

    def test_main_with_everything_skipped(self):
        argv = ["-cd", self.case_dir, "-cnp", self.require_fixture("cta.nii.gz")] + ["--" + flag for flag in SKIP_FLAGS]
        times = perform_analysis.main(argv)
        self.assertIn("total_time", times)

    def test_readme_option_table_matches_parser(self):
        parser = perform_analysis.build_parser()
        parser_flags = {opt for action in parser._actions for opt in action.option_strings if opt not in ("-h", "--help")}
        with open(os.path.join(REPO_ROOT, "README.md"), encoding="utf-8") as handle:
            readme = handle.read()
        table = readme.split("## Command-Line Options", 1)[1].split("\n## ", 1)[0]
        documented = set(re.findall(r"`(-{1,2}[A-Za-z0-9_]+)`", table))
        self.assertEqual(parser_flags - documented, set(), "flags missing from the README table")
        self.assertEqual(documented - parser_flags, set(), "README documents flags the parser does not have")


class TestFullPipelineExample(ArterialTestCase):

    def test_params_file_maps_to_processor_namespace(self):
        with open(os.path.join(REPO_ROOT, "example_scripts", "arterial_processing_params.json"), encoding="utf-8") as handle:
            params = json.load(handle)
        params["case_dir"] = self.case_dir
        params["cta_nifti_path"] = ""  # empty means case_dir/cta.nii.gz
        params["skip_branching"] = True
        params["skip_clipping"] = False
        args = arterial_processing_full_pipeline.load_args_from_params_json(params)
        self.assertTrue(args.skip_branching)
        self.assertFalse(args.skip_clipping)
        self.assertEqual(args.cta_nifti_path, os.path.join(self.case_dir, "cta.nii.gz"))
        processor = ArterialProcessor(args)
        self.assertEqual(processor.case_dir, self.case_dir)

    def test_params_without_case_dir_is_rejected(self):
        with self.assertRaises(ValueError):
            arterial_processing_full_pipeline.load_args_from_params_json({})
