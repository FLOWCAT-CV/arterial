#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
import subprocess
import sys
import unittest
from unittest import mock

from arterial import model_registry


class TestImportSideEffects(unittest.TestCase):
    """Importing the package must not write to stdout: subprocess workers return pickles over it."""

    def test_banner_goes_to_stderr_not_stdout(self):
        result = subprocess.run([sys.executable, "-c", "import arterial"], capture_output=True, text=True, timeout=120,
                                env={**os.environ, "PYTHONUNBUFFERED": "1"})
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "", "import arterial wrote to stdout")
        self.assertIn("PolyForm Noncommercial", result.stderr)
        self.assertIn("Universitat de Barcelona", result.stderr)


class TestModelRegistry(unittest.TestCase):
    """Resolution order of the models directory: ARTERIAL_MODELS_DIR, else $arterial_dir/models."""

    def test_env_override_takes_precedence(self):
        with mock.patch.dict(os.environ, {"ARTERIAL_MODELS_DIR": "/data/models", "arterial_dir": "/repo/arterial"}):
            self.assertEqual(model_registry.models_dir(), "/data/models")
            self.assertEqual(model_registry.model_path("landmark_detection", "x.pth"), "/data/models/landmark_detection/x.pth")

    def test_falls_back_to_arterial_dir(self):
        with mock.patch.dict(os.environ, {"arterial_dir": "/repo/arterial"}, clear=False):
            os.environ.pop("ARTERIAL_MODELS_DIR", None)
            self.assertEqual(model_registry.models_dir(), "/repo/arterial/models")

    def test_raises_when_nothing_is_set(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("ARTERIAL_MODELS_DIR", None)
            os.environ.pop("arterial_dir", None)
            with self.assertRaises(EnvironmentError):
                model_registry.models_dir()

    def test_configured_directory_holds_the_published_layout(self):
        models_dir = model_registry.models_dir()
        if not os.path.isdir(models_dir):
            self.skipTest(f"models not installed at {models_dir}")
        for parts in [("landmark_detection", "six_landmarks_2ch.pth"), ("access_prediction", "dataset.json"),
                      ("vessel_labelling", "extracranial_vessels", "model_weights.pth"),
                      ("segmentation", "totalsegmentator_mandible", "LICENSE"),
                      ("segmentation", "totalsegmentator_mandible", "NOTICE")]:
            with self.subTest(path="/".join(parts)):
                self.assertTrue(os.path.exists(model_registry.model_path(*parts)))
