#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

from types import SimpleNamespace
from unittest import mock

import numpy as np
import torch
from nnunetv2.preprocessing.resampling.default_resampling import resample_data_or_seg_to_shape
from nnunetv2.preprocessing.resampling.resample_torch import resample_torch_fornnunet
from nnunetv2.utilities.plans_handling.plans_handler import ConfigurationManager

from helpers import ArterialTestCase
from arterial.segmentation import inference


def plans_configuration():
    # Only the entries of a plans.json configuration that ConfigurationManager needs here
    configuration = {"architecture": {}}
    for fn, is_seg in [("resampling_fn_data", False), ("resampling_fn_seg", True), ("resampling_fn_probabilities", False)]:
        configuration[fn] = "resample_data_or_seg_to_shape"
        configuration[f"{fn}_kwargs"] = {"is_seg": is_seg, "order": 1 if is_seg else 3, "order_z": 0, "force_separate_z": None}
    return configuration


def fake_predictor(device="cpu"):
    return SimpleNamespace(configuration_manager=ConfigurationManager(plans_configuration()), device=torch.device(device))


class TestPredictorCache(ArterialTestCase):

    def setUp(self):
        super().setUp()
        inference._PREDICTORS.clear()
        self.addCleanup(inference._PREDICTORS.clear)

    def test_predictor_is_built_once_per_model_folds_and_device(self):
        with mock.patch.object(inference, "nnUNetPredictor", side_effect=lambda **kwargs: mock.MagicMock()) as predictor_class:
            first = inference.get_nnunet_predictor("/models/a", ("all",), torch.device("cpu"))
            second = inference.get_nnunet_predictor("/models/a", ("all",), torch.device("cpu"))
            other_folds = inference.get_nnunet_predictor("/models/a", ("0",), torch.device("cpu"))
        self.assertIs(first, second)
        self.assertIsNot(first, other_folds)
        self.assertEqual(predictor_class.call_count, 2)
        first.initialize_from_trained_model_folder.assert_called_once_with("/models/a", use_folds=("all",), checkpoint_name="checkpoint_final.pth")


class TestResamplingSwitch(ArterialTestCase):

    def test_switch_between_torch_and_plans_resampling(self):
        predictor = fake_predictor()
        configuration_manager = predictor.configuration_manager
        self.assertIs(configuration_manager.resampling_fn_data.func, resample_data_or_seg_to_shape)

        inference.set_nnunet_resampling(predictor, "torch", torch.device("cpu"), num_threads=2)
        for fn in [configuration_manager.resampling_fn_data, configuration_manager.resampling_fn_seg, configuration_manager.resampling_fn_probabilities]:
            self.assertIs(fn.func, resample_torch_fornnunet)
            self.assertEqual(fn.keywords["num_threads"], 2)
        self.assertTrue(configuration_manager.resampling_fn_seg.keywords["is_seg"])

        inference.set_nnunet_resampling(predictor, "scipy")
        self.assertIs(configuration_manager.resampling_fn_data.func, resample_data_or_seg_to_shape)
        self.assertEqual(configuration_manager.configuration, plans_configuration())

    def test_both_resamplings_return_the_target_grid(self):
        data = np.random.default_rng(0).random((1, 10, 12, 14)).astype(np.float32)
        for resampling in ["scipy", "torch"]:
            predictor = fake_predictor()
            inference.set_nnunet_resampling(predictor, resampling, torch.device("cpu"), num_threads=1)
            resampled = predictor.configuration_manager.resampling_fn_probabilities(data, [20, 18, 7], [1.0, 1.0, 1.0], [0.5, 0.67, 2.0])
            self.assertEqual(tuple(resampled.shape), (1, 20, 18, 7), resampling)

    def test_unknown_resampling_is_rejected(self):
        with self.assertRaises(AssertionError):
            inference.set_nnunet_resampling(fake_predictor(), "cubic")


class TestRunPrediction(ArterialTestCase):

    def test_torch_resampling_needs_cuda(self):
        predictor = fake_predictor("cpu")
        predictor.predict_single_npy_array = mock.MagicMock(return_value="segmentation")
        self.assertEqual(inference.run_nnunet_prediction(predictor, None, {}, False, "torch"), "segmentation")
        self.assertIs(predictor.configuration_manager.resampling_fn_data.func, resample_data_or_seg_to_shape)

    def test_out_of_memory_falls_back_to_scipy(self):
        predictor = fake_predictor("cuda")
        used_resampling = []
        def predict(*args):
            used_resampling.append(predictor.configuration_manager.resampling_fn_data.func)
            if len(used_resampling) == 1:
                raise torch.cuda.OutOfMemoryError("out of memory")
            return "segmentation"
        predictor.predict_single_npy_array = predict
        with mock.patch.object(torch.cuda, "empty_cache"):
            self.assertEqual(inference.run_nnunet_prediction(predictor, None, {}, False, "torch", 1), "segmentation")
        self.assertEqual(used_resampling, [resample_torch_fornnunet, resample_data_or_seg_to_shape])
