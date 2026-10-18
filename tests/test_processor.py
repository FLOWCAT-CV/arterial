#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import argparse
import json
import os
import re
import subprocess
import sys
from unittest import mock

from helpers import ArterialTestCase
from arterial.run import processor as processor_module
from arterial.run.processor import ArterialProcessor, SUPPORTED_MODES
from arterial import cli

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "tutorials", "scripts"))
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
    """The arterial command line and the README's option table."""

    def test_parser_defaults_match_processor_expectations(self):
        args = cli.build_parser().parse_args(["-cd", self.case_dir])
        for flag in SKIP_FLAGS + ["fast_segmentation", "cl_dice_nnunet", "no_slicing", "set_threshold_099"]:
            self.assertFalse(getattr(args, flag), flag)
        self.assertEqual(args.mode, "extracranial_vessels")
        self.assertEqual(args.sampling_distance_mm, 2)

    def test_parser_rejects_unknown_mode(self):
        with self.assertRaises(SystemExit):
            cli.build_parser().parse_args(["-cd", self.case_dir, "-m", "thrombus"])
        for mode in SUPPORTED_MODES:
            self.assertEqual(cli.build_parser().parse_args(["-cd", self.case_dir, "-m", mode]).mode, mode)

    def test_help_runs_as_a_module(self):
        result = subprocess.run([sys.executable, "-m", "arterial.cli", "--help"],
                                capture_output=True, text=True, timeout=300)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--skip_segmentation", result.stdout)

    def test_run_with_everything_skipped(self):
        argv = ["-cd", self.case_dir, "-cnp", self.require_fixture("cta.nii.gz")] + ["--" + flag for flag in SKIP_FLAGS]
        times = cli.run(argv)
        self.assertIn("total_time", times)
        with open(os.path.join(self.mode_dir, "arterial_run.json"), encoding="utf-8") as handle:
            record = json.load(handle)
        self.assertEqual(record["settings"]["resampling"], "torch")
        self.assertTrue(all(record["settings"][flag] for flag in processor_module.SKIP_FLAGS))

    def test_cta_path_needs_a_single_case(self):
        with mock.patch("sys.stderr"), self.assertRaises(SystemExit):
            cli.run(["-cd", self.case_dir, os.path.join(self.case_dir, "other"), "-cnp", "cta.nii.gz"])

    def test_command_exit_status(self):
        # The console script calls sys.exit(main()): a successful run must exit with 0
        argv = ["-cd", self.case_dir, "-cnp", self.require_fixture("cta.nii.gz")] + ["--" + flag for flag in SKIP_FLAGS]
        result = subprocess.run([sys.executable, "-c", "import sys; from arterial.cli import main; sys.exit(main(sys.argv[1:]))"] + argv,
                                capture_output=True, text=True, timeout=300, cwd=REPO_ROOT)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])

    def test_exit_status_reports_failed_cases_and_incomplete_analyses(self):
        argv = ["-cd", self.case_dir, os.path.join(self.case_dir, "other")]
        with mock.patch.object(cli, "process_cases", return_value={self.case_dir: {"total_time": 1.0}}):
            self.assertEqual(cli.main(argv), 0)
        with mock.patch.object(cli, "process_cases", return_value={self.case_dir: {"total_time": 1.0}, "other": "error: broken"}):
            self.assertEqual(cli.main(argv), 1)
        with mock.patch.object(cli, "process_single_case", side_effect=processor_module.IncompleteAnalysisError("missing landmarks.json")):
            self.assertEqual(cli.main(["-cd", self.case_dir]), 1)

    def test_readme_option_table_matches_parser(self):
        parser = cli.build_parser()
        parser_flags = {opt for action in parser._actions for opt in action.option_strings if opt not in ("-h", "--help")}
        with open(os.path.join(REPO_ROOT, "README.md"), encoding="utf-8") as handle:
            readme = handle.read()
        table = readme.split("## Command-Line Options", 1)[1].split("\n## ", 1)[0]
        documented = set(re.findall(r"`(-{1,2}[A-Za-z0-9_]+)`", table))
        self.assertEqual(parser_flags - documented, set(), "flags missing from the README table")
        self.assertEqual(documented - parser_flags, set(), "README documents flags the parser does not have")


class TestMultipleCases(ArterialTestCase):
    """Several case directories in one run, without running any stage."""

    def setUp(self):
        super().setUp()
        self.case_dirs = [os.path.join(self.case_dir, name) for name in ["case_a", "case_b"]]
        # Never query a GPU from the fast tier
        patcher = mock.patch.object(processor_module.torch.cuda, "is_available", return_value=False)
        patcher.start()
        self.addCleanup(patcher.stop)

    def args(self, skip=SKIP_FLAGS, **kwargs):
        return cli.build_parser().parse_args(["-cd"] + self.case_dirs + ["--" + flag for flag in skip] + sum(([f"--{k}", str(v)] for k, v in kwargs.items()), []))

    def single_case_args(self, case_dir, args):
        return argparse.Namespace(**{**vars(args), "case_dir": case_dir})

    def write_run_record(self, case_dir, args):
        os.makedirs(os.path.join(case_dir, "extracranial_vessels"), exist_ok=True)
        with open(os.path.join(case_dir, "extracranial_vessels", "arterial_run.json"), "w", encoding="utf-8") as handle:
            json.dump({"settings": processor_module.get_run_settings(self.single_case_args(case_dir, args))}, handle)

    def test_completed_cases_are_skipped(self):
        args = self.args(num_threads=4)
        self.write_run_record(self.case_dirs[0], args)
        with mock.patch.object(processor_module, "process_single_case", return_value={"total_time": 0}) as process:
            results = processor_module.process_cases(args)
        self.assertEqual(list(results), [self.case_dirs[1]])
        self.assertEqual(process.call_args[0][0].case_dir, self.case_dirs[1])
        self.assertEqual(process.call_args[0][0].num_threads, 4)

    def test_case_completed_with_other_settings_is_processed_again(self):
        self.write_run_record(self.case_dirs[0], self.args(resampling="scipy"))
        with mock.patch.object(processor_module, "process_single_case", return_value={"total_time": 0}):
            results = processor_module.process_cases(self.args())
        self.assertEqual(sorted(results), self.case_dirs)

    def test_case_completed_on_another_image_is_processed_again(self):
        args = self.args()
        record_args = self.single_case_args(self.case_dirs[0], args)
        record_args.cta_nifti_path = os.path.join(self.case_dir, "another_cta.nii.gz")
        os.makedirs(os.path.join(self.case_dirs[0], "extracranial_vessels"))
        with open(os.path.join(self.case_dirs[0], "extracranial_vessels", "arterial_run.json"), "w", encoding="utf-8") as handle:
            json.dump({"settings": processor_module.get_run_settings(record_args)}, handle)
        self.assertFalse(processor_module.is_case_processed(self.single_case_args(self.case_dirs[0], args)))

    def test_case_whose_outputs_were_deleted_is_processed_again(self):
        args = self.args(skip=[flag for flag in SKIP_FLAGS if flag != "skip_segmentation"])
        self.write_run_record(self.case_dirs[0], args)
        segmentation_path = os.path.join(self.case_dirs[0], "extracranial_vessels", "segmentation.nii.gz")
        open(segmentation_path, "w").close()
        self.assertTrue(processor_module.is_case_processed(self.single_case_args(self.case_dirs[0], args)))
        os.remove(segmentation_path)
        self.assertFalse(processor_module.is_case_processed(self.single_case_args(self.case_dirs[0], args)))

    def test_duplicate_case_directories_are_processed_once(self):
        args = self.args()
        args.case_dir = [self.case_dirs[0], self.case_dirs[0] + os.sep, self.case_dirs[1]]
        with mock.patch.object(processor_module, "process_single_case", return_value={"total_time": 0}) as process:
            results = processor_module.process_cases(args)
        self.assertEqual(process.call_count, 2)
        self.assertEqual(list(results), self.case_dirs)

    def test_incomplete_case_is_not_marked_as_done(self):
        args = self.single_case_args(self.case_dirs[0], self.args(skip=[flag for flag in SKIP_FLAGS if flag != "skip_vessel_labelling"]))
        with mock.patch.object(ArterialProcessor, "perform_analysis", return_value={"total_time": 0}):
            with self.assertRaisesRegex(processor_module.IncompleteAnalysisError, "segments_graph_pred.pickle"):
                processor_module.process_single_case(args)
        self.assertFalse(processor_module.is_case_processed(args))

    def test_failing_rerun_removes_the_earlier_run_record(self):
        args = self.single_case_args(self.case_dirs[0], self.args())
        self.write_run_record(self.case_dirs[0], self.args())
        self.assertTrue(processor_module.is_case_processed(args))
        with mock.patch.object(ArterialProcessor, "perform_analysis", side_effect=RuntimeError("stage failed")):
            with self.assertRaises(RuntimeError):
                processor_module.process_single_case(args)
        self.assertFalse(processor_module.is_case_processed(args))

    def test_completeness_covers_every_stage_that_ran(self):
        for mode in ["extracranial_vessels", "intracranial_vessels"]:
            with self.subTest(mode=mode):
                processor = ArterialProcessor(argparse.Namespace(case_dir=self.case_dirs[0], mode=mode))
                missing = {os.path.relpath(path, os.path.join(self.case_dirs[0], mode)) for path in processor_module.get_missing_outputs(processor)}
                expected = {"segmentation.nii.gz", "branch_model.vtk", "landmarks.json", "segments_graph_pred.pickle"}
                if mode == "extracranial_vessels":
                    expected |= {"local_graph.pickle", os.path.join("access_prediction", "femoral_left", "access_prediction.json"),
                                 os.path.join("access_prediction", "femoral_right", "access_prediction.json")}
                self.assertEqual(missing, expected)

    def test_failed_case_does_not_stop_the_run(self):
        with mock.patch.object(processor_module, "process_single_case", side_effect=[RuntimeError("broken"), {"total_time": 0}]):
            results = processor_module.process_cases(self.args())
        self.assertEqual(results[self.case_dirs[0]], "error: broken")
        self.assertEqual(results[self.case_dirs[1]], {"total_time": 0})

    def test_crash_is_attributed_only_to_the_case_that_caused_it(self):
        submitted = []
        class FakeExecutor:
            # The shared pool breaks for every case when one worker dies; in a pool of its own only case_a crashes
            def __init__(self, max_workers, mp_context):
                self.shared = max_workers > 1
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def submit(self, fn, case_args, workers):
                submitted.append((self.shared, case_args.case_dir))
                future = mock.MagicMock()
                crashes = self.shared or case_args.case_dir.endswith("case_a")
                future.result.side_effect = processor_module.BrokenProcessPool("worker died") if crashes else None
                future.result.return_value = {"total_time": 0}
                return future
        with mock.patch.object(processor_module, "get_max_workers", return_value=2), \
             mock.patch.object(processor_module, "ProcessPoolExecutor", FakeExecutor), \
             mock.patch.object(processor_module, "as_completed", side_effect=lambda futures: list(futures)), \
             mock.patch.dict(os.environ):
            results = processor_module.process_cases(self.args())
        self.assertTrue(results[self.case_dirs[0]].startswith("error: "))
        self.assertEqual(results[self.case_dirs[1]], {"total_time": 0})
        self.assertEqual(submitted[2:], [(False, self.case_dirs[0]), (False, self.case_dirs[1])])

    def test_default_threads_are_capped_per_worker(self):
        # 64 usable cores and 2 workers: each worker gets MAX_NUM_THREADS, not MAX_NUM_THREADS // 2
        with mock.patch.object(processor_module, "get_max_workers", return_value=2), \
             mock.patch.object(processor_module, "get_num_threads", return_value=(processor_module.MAX_NUM_THREADS, 64, "default")), \
             mock.patch.object(processor_module, "ProcessPoolExecutor", side_effect=RuntimeError("stop before starting workers")), \
             mock.patch.dict(os.environ):
            os.environ.pop("OMP_NUM_THREADS", None)
            with self.assertRaises(RuntimeError):
                processor_module.process_cases(self.args())
            self.assertEqual(os.environ["OMP_NUM_THREADS"], str(processor_module.MAX_NUM_THREADS))

    def test_workers_are_limited_by_cases_gpu_memory_and_ram(self):
        self.assertEqual(processor_module.get_max_workers(4, 2), 2)
        self.assertEqual(processor_module.get_max_workers(1, 10), 1)
        with mock.patch.object(processor_module.torch.cuda, "is_available", return_value=True), \
             mock.patch.object(processor_module.torch.cuda, "mem_get_info", return_value=(2.5 * processor_module.GPU_MEMORY_PER_WORKER_GB * 1024 ** 3, 0)), \
             mock.patch.object(processor_module, "get_available_ram_gb", return_value=1000):
            self.assertEqual(processor_module.get_max_workers(4, 10), 2)
        with mock.patch.object(processor_module, "get_available_ram_gb", return_value=2.5 * processor_module.RAM_PER_WORKER_GB):
            self.assertEqual(processor_module.get_max_workers(4, 10), 2)
        with mock.patch.object(processor_module, "get_available_ram_gb", return_value=1):
            self.assertEqual(processor_module.get_max_workers(4, 10), 1)


class TestFullPipelineExample(ArterialTestCase):

    def test_params_file_maps_to_processor_namespace(self):
        with open(os.path.join(REPO_ROOT, "tutorials", "scripts", "arterial_processing_params.json"), encoding="utf-8") as handle:
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
