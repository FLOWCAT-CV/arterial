#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
scripts/download_models.sh against a local HTTP server that mimics Zenodo's
records/<id>/files/<name> layout with a tiny fake archive.
"""

import hashlib
import http.server
import io
import os
import shutil
import subprocess
import tarfile
import threading
import unittest

from helpers import ArterialTestCase

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO_ROOT, "scripts", "download_models.sh")
ARCHIVE = "arterial-models-v1.tar.gz"

REQUIRED_FILES = [
    "access_prediction/dataset.json",
    "access_prediction/fold_0/model_weights.pth",
    "landmark_detection/six_landmarks_2ch.pth",
    "landmark_detection/six_landmarks_11_7.pth",
    "segmentation/extracranial_vessels/nnUNetTrainer__nnUNetPlans__3d_lowres/plans.json",
    "segmentation/intracranial_vessels/nnUNetTrainer__nnUNetPlans__3d_fullres/plans.json",
    "segmentation/totalsegmentator_mandible/nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres/plans.json",
    "segmentation/totalsegmentator_mandible/LICENSE",
    "segmentation/totalsegmentator_mandible/NOTICE",
    "vessel_labelling/extracranial_vessels/dataset.json",
    "vessel_labelling/extracranial_vessels/model_weights.pth",
]
EXTRA_CHECKPOINTS = [f"access_prediction/fold_{i}/model_weights.pth" for i in range(1, 5)] + \
    [f"vessel_labelling/extracranial_vessels/fold_{i}/model_weights.pth" for i in range(5)] + \
    ["segmentation/extracranial_vessels/nnUNetTrainer__nnUNetPlans__3d_lowres/fold_0/checkpoint_final.pth",
     "segmentation/extracranial_vessels/nnUNetTrainer__nnUNetPlans__3d_lowres/fold_all/checkpoint_final.pth",
     "segmentation/intracranial_vessels/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth",
     "segmentation/intracranial_vessels/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_all/checkpoint_final.pth",
     "segmentation/totalsegmentator_mandible/nnUNetTrainer_DASegOrd0_NoMirroring__nnUNetPlans__3d_fullres/fold_0/checkpoint_final.pth"]


def build_archive(members, sidecars=()):
    """
    Builds an in-memory tar.gz with the given member paths (tiny contents).

    """
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in list(members) + list(sidecars):
            data = b"x" * 8
            info = tarfile.TarInfo(name="./" + name)
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


class FakeZenodo:
    """
    Serves records/<id>/files/<name> from a dict; a name mapped to an int is
    answered with that HTTP status.

    """

    def __init__(self, files):
        self.files = files
        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_GET(self):
                name = self.path.split("/files/", 1)[-1].split("?", 1)[0]
                entry = outer.files.get(name)
                if entry is None:
                    self.send_response(404); self.end_headers(); return
                if isinstance(entry, int):
                    self.send_response(entry); self.end_headers(); return
                self.send_response(200); self.send_header("Content-Length", str(len(entry))); self.end_headers()
                self.wfile.write(entry)

            def log_message(self, *args):
                pass

        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self):
        self.thread.start()
        return f"http://127.0.0.1:{self.server.server_address[1]}"

    def __exit__(self, *exc):
        self.server.shutdown()
        self.server.server_close()


@unittest.skipUnless(shutil.which("bash") and shutil.which("curl") and shutil.which("tar"), "bash, curl and tar are required")
class TestDownloadScript(ArterialTestCase):

    def setUp(self):
        super().setUp()
        self.home = os.path.join(self.case_dir, "home"); os.makedirs(self.home)
        self.dest = os.path.join(self.case_dir, "models")
        self.archive = build_archive(REQUIRED_FILES + EXTRA_CHECKPOINTS)
        self.sha256 = (hashlib.sha256(self.archive).hexdigest() + "  " + ARCHIVE + "\n").encode()

    def run_script(self, site, *args, persist=False, env_extra=None):
        env = {"PATH": os.environ["PATH"], "HOME": self.home, "SHELL": "/bin/bash", "ARTERIAL_MODELS_DIR": self.dest}
        env.update(env_extra or {})
        argv = ["bash", SCRIPT, "--site", site, "--record", "1", *args]
        if not persist:
            argv.append("--no-persist")
        return subprocess.run(argv, capture_output=True, text=True, env=env, timeout=120)

    def files(self, **overrides):
        files = {ARCHIVE: self.archive, ARCHIVE + ".sha256": self.sha256}
        files.update(overrides)
        return files

    def test_happy_path_installs_and_verifies(self):
        with FakeZenodo(self.files()) as site:
            result = self.run_script(site)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("Checksum OK.", result.stdout)
        self.assertIn("checkpoints found: 18 (expected 18)", result.stdout)
        for name in REQUIRED_FILES:
            self.assertFileExists(os.path.join(self.dest, name))
        self.assertFileMissing(os.path.join(self.home, ".bashrc"))

    def test_checksum_mismatch_fails(self):
        with FakeZenodo(self.files(**{ARCHIVE + ".sha256": ("0" * 64 + "  x\n").encode()})) as site:
            result = self.run_script(site)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Checksum mismatch", result.stderr)

    def test_server_error_on_checksum_fails_closed(self):
        with FakeZenodo(self.files(**{ARCHIVE + ".sha256": 503})) as site:
            result = self.run_script(site)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("HTTP 503", result.stderr)
        self.assertFalse(os.path.exists(self.dest), "nothing must be extracted without verification")

    def test_missing_checksum_is_skipped_with_a_notice(self):
        files = self.files(); del files[ARCHIVE + ".sha256"]
        with FakeZenodo(files) as site:
            result = self.run_script(site)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("HTTP 404", result.stdout)

    def test_missing_archive_fails(self):
        with FakeZenodo({}) as site:
            result = self.run_script(site)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Download failed", result.stderr)

    def test_apple_double_sidecars_are_excluded(self):
        sidecars = ["._." ] + ["landmark_detection/._six_landmarks_2ch.pth", "access_prediction/fold_0/._model_weights.pth", "segmentation/totalsegmentator_mandible/._LICENSE"]
        archive = build_archive(REQUIRED_FILES + EXTRA_CHECKPOINTS, sidecars)
        sha = (hashlib.sha256(archive).hexdigest() + "  " + ARCHIVE + "\n").encode()
        with FakeZenodo({ARCHIVE: archive, ARCHIVE + ".sha256": sha}) as site:
            result = self.run_script(site)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("checkpoints found: 18 (expected 18)", result.stdout)
        leftovers = [os.path.join(root, f) for root, _, fs in os.walk(self.dest) for f in fs if f.startswith("._")]
        self.assertEqual(leftovers, [])

    def test_incomplete_archive_fails_verification(self):
        archive = build_archive(REQUIRED_FILES)  # 4 checkpoints instead of 18
        sha = (hashlib.sha256(archive).hexdigest() + "  " + ARCHIVE + "\n").encode()
        with FakeZenodo({ARCHIVE: archive, ARCHIVE + ".sha256": sha}) as site:
            result = self.run_script(site)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("Verification failed", result.stderr)

    def test_invalid_record_and_unknown_flag(self):
        env = {"PATH": os.environ["PATH"], "HOME": self.home, "ARTERIAL_MODELS_DIR": self.dest}
        bad_record = subprocess.run(["bash", SCRIPT, "--record", "abc", "--no-persist"], capture_output=True, text=True, env=env, timeout=60)
        self.assertNotEqual(bad_record.returncode, 0)
        self.assertIn("Invalid Zenodo record id", bad_record.stderr)
        unknown = subprocess.run(["bash", SCRIPT, "--bogus"], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(unknown.returncode, 2)
        self.assertIn("Usage", unknown.stderr)
        helptext = subprocess.run(["bash", SCRIPT, "--help"], capture_output=True, text=True, env=env, timeout=60)
        self.assertEqual(helptext.returncode, 0)
        self.assertIn("--site", helptext.stdout)
        self.assertIn("22694951", helptext.stdout)

    def test_persist_writes_one_marked_block_and_is_idempotent(self):
        rc = os.path.join(self.home, ".bashrc")
        with open(rc, "w") as handle:
            handle.write("# keep me\nexport PATH=$PATH\n")
        os.chmod(rc, 0o644)
        with FakeZenodo(self.files()) as site:
            first = self.run_script(site, persist=True)
            second = self.run_script(site, persist=True)
        self.assertEqual(first.returncode, 0, first.stderr)
        self.assertEqual(second.returncode, 0, second.stderr)
        with open(rc) as handle:
            content = handle.read()
        self.assertEqual(content.count("# >>> arterial models >>>"), 1)
        self.assertIn(f'export ARTERIAL_MODELS_DIR="{self.dest}"', content)
        self.assertTrue(content.startswith("# keep me\nexport PATH=$PATH\n"))
        self.assertEqual(os.stat(rc).st_mode & 0o777, 0o644, "the rc file must keep its permissions")

    def test_default_destination_is_arterial_dir_models(self):
        package_dir = os.path.join(self.case_dir, "pkg"); os.makedirs(package_dir)
        env = {"PATH": os.environ["PATH"], "HOME": self.home, "arterial_dir": package_dir}
        with FakeZenodo(self.files()) as site:
            result = subprocess.run(["bash", SCRIPT, "--site", site, "--record", "1", "--no-persist"], capture_output=True, text=True, env=env, timeout=120)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFileExists(os.path.join(package_dir, "models", "landmark_detection", "six_landmarks_2ch.pth"))
