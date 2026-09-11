#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Shared scaffolding for the Arterial test suite.

Every test gets its own temporary case directory, so tests never share
state through the filesystem and can run in any order. Input fixtures live
in ``tests/test_data/input_test_data`` (gitignored, see tests/README.md).

Tiers
-----
Tests that run model inference or VMTK are decorated with ``@slow``. They
run by default; set ``ARTERIAL_SKIP_SLOW=1`` to run only the fast tier.
"""

import os
import shutil
import tempfile
import unittest

import numpy as np

FIXTURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_data", "input_test_data")

SKIP_SLOW = os.environ.get("ARTERIAL_SKIP_SLOW", "") == "1"

slow = unittest.skipIf(SKIP_SLOW, "slow test skipped (ARTERIAL_SKIP_SLOW=1)")


def fixture(*parts):
    """
    Builds the path to an input fixture.

    Parameters
    ----------
    *parts : str
        Path components relative to the fixtures directory.

    Returns
    -------
    path : str
        Absolute path to the fixture.

    """
    return os.path.join(FIXTURES_DIR, *parts)


class ArterialTestCase(unittest.TestCase):
    """
    Base class giving each test a fresh temporary case directory.

    Attributes
    ----------
    case_dir : str
        Temporary directory removed after the test, whatever its outcome.
    mode : str
        Processing mode, ``extracranial_vessels`` unless a subclass overrides it.

    """
    mode = "extracranial_vessels"

    def setUp(self):
        self.case_dir = tempfile.mkdtemp(prefix="arterial_test_")
        self.addCleanup(shutil.rmtree, self.case_dir, ignore_errors=True)

    @property
    def mode_dir(self):
        """
        Returns the per-mode output directory inside the case directory.

        """
        return os.path.join(self.case_dir, self.mode)

    def require_fixture(self, *parts):
        """
        Skips the test when an input fixture is missing, and returns its path.

        Parameters
        ----------
        *parts : str
            Path components relative to the fixtures directory.

        Returns
        -------
        path : str
            Absolute path to the fixture.

        """
        path = fixture(*parts)
        if not os.path.exists(path):
            self.skipTest(f"missing fixture input_test_data/{'/'.join(parts)}")
        return path

    def stage_fixture(self, name, *parts):
        """
        Copies a fixture into the per-mode output directory, as if a previous
        pipeline stage had produced it.

        Parameters
        ----------
        name : str
            Filename to give the copy inside the mode directory.
        *parts : str
            Path components of the fixture relative to the fixtures directory.

        Returns
        -------
        path : str
            Absolute path to the copy.

        """
        src = self.require_fixture(*parts)
        os.makedirs(self.mode_dir, exist_ok=True)
        dst = os.path.join(self.mode_dir, name)
        if os.path.isdir(src):
            shutil.copytree(src, dst)
        else:
            shutil.copy(src, dst)
        return dst

    def assertFileExists(self, path):
        self.assertTrue(os.path.isfile(path), f"expected file is missing: {path}")

    def assertFileMissing(self, path):
        self.assertFalse(os.path.exists(path), f"file should not exist: {path}")

    def assertDirExists(self, path):
        self.assertTrue(os.path.isdir(path), f"expected directory is missing: {path}")


def ras_bounding_box(nifti):
    """
    Computes the RAS bounding box of a nifti volume from its affine.

    Parameters
    ----------
    nifti : nibabel image
        Volume whose corners are transformed.

    Returns
    -------
    lower, upper : numpy.ndarray
        Minimum and maximum RAS coordinates of the volume, in mm.

    """
    shape = np.array(nifti.shape[:3]) - 1
    corners = np.array([[i, j, k, 1] for i in (0, shape[0]) for j in (0, shape[1]) for k in (0, shape[2])], dtype=float)
    ras = (nifti.affine @ corners.T).T[:, :3]
    return ras.min(axis=0), ras.max(axis=0)
