#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
from unittest import mock

import numpy as np
import nibabel as nib
from skimage.measure import label

from helpers import ArterialTestCase, slow
from arterial.segmentation import utils as seg_utils


class TestConnectedComponents(ArterialTestCase):

    def test_largest_component_is_kept(self):
        volume = np.zeros((12, 12, 12), dtype=np.uint8)
        volume[1:4, 1:4, 1:4] = 1   # 27 voxels
        volume[8:10, 8:10, 8:10] = 1  # 8 voxels
        largest = seg_utils.get_largest_connected_component(volume)
        self.assertEqual(int(np.count_nonzero(largest)), 27)
        self.assertTrue(largest[2, 2, 2])
        self.assertFalse(largest[9, 9, 9])

    def test_empty_volume_is_rejected(self):
        with self.assertRaises(AssertionError):
            seg_utils.get_largest_connected_component(np.zeros((4, 4, 4), dtype=np.uint8))

    def test_same_component_as_skimage_label(self):
        # Blobs that only touch diagonally are one component with 26-connectivity
        volume = np.random.default_rng(0).random((40, 40, 40)) > 0.75
        labels = label(volume)
        expected = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
        np.testing.assert_array_equal(seg_utils.get_largest_connected_component(volume), expected)


class TestNumThreads(ArterialTestCase):

    def setUp(self):
        super().setUp()
        # No cgroup quota unless a test writes one
        patcher = mock.patch.multiple(seg_utils, CGROUP_CPU_MAX_PATH=os.path.join(self.case_dir, "cpu.max"),
                                      CGROUP_V1_CPU_DIR=os.path.join(self.case_dir, "cpu"))
        patcher.start()
        self.addCleanup(patcher.stop)
        patcher = mock.patch.dict(os.environ)
        patcher.start()
        self.addCleanup(patcher.stop)
        os.environ.pop("ARTERIAL_NUM_THREADS", None)

    def test_default_is_usable_cores_capped(self):
        with mock.patch("os.sched_getaffinity", return_value=set(range(28)), create=True):
            self.assertEqual(seg_utils.get_num_threads(), (seg_utils.MAX_NUM_THREADS, 28, "default"))
        with mock.patch("os.sched_getaffinity", return_value={0, 1, 2}, create=True):
            self.assertEqual(seg_utils.get_num_threads(), (3, 3, "default"))

    def test_cgroup_v2_quota_limits_usable_cores(self):
        with open(seg_utils.CGROUP_CPU_MAX_PATH, "w") as f:
            f.write("250000 100000\n")
        with mock.patch("os.sched_getaffinity", return_value=set(range(28)), create=True):
            self.assertEqual(seg_utils.get_usable_cores(), 3)
        with open(seg_utils.CGROUP_CPU_MAX_PATH, "w") as f:
            f.write("max 100000\n")
        with mock.patch("os.sched_getaffinity", return_value=set(range(28)), create=True):
            self.assertEqual(seg_utils.get_usable_cores(), 28)

    def test_cgroup_v1_quota_limits_usable_cores(self):
        os.makedirs(seg_utils.CGROUP_V1_CPU_DIR)
        with open(os.path.join(seg_utils.CGROUP_V1_CPU_DIR, "cpu.cfs_quota_us"), "w") as f:
            f.write("200000\n")
        with open(os.path.join(seg_utils.CGROUP_V1_CPU_DIR, "cpu.cfs_period_us"), "w") as f:
            f.write("100000\n")
        with mock.patch("os.sched_getaffinity", return_value=set(range(28)), create=True):
            self.assertEqual(seg_utils.get_usable_cores(), 2)

    def test_without_affinity_cpu_count_is_used(self):
        with mock.patch("os.sched_getaffinity", side_effect=AttributeError, create=True), mock.patch("os.cpu_count", return_value=6):
            self.assertEqual(seg_utils.get_usable_cores(), 6)

    def test_argument_and_environment_variable_take_precedence(self):
        os.environ["ARTERIAL_NUM_THREADS"] = "12"
        self.assertEqual(seg_utils.get_num_threads()[::2], (12, "ARTERIAL_NUM_THREADS"))
        self.assertEqual(seg_utils.get_num_threads(2)[::2], (2, "argument"))
        self.assertEqual(seg_utils.get_num_threads(0)[0], 1)


class TestCraniumMask(ArterialTestCase):

    def _cta(self, dtype):
        cta = np.zeros((32, 32, 40), dtype=dtype)
        cta[8:24, 8:24, 24:30] = 1000  # a bright slab in the upper half
        return cta

    def test_float_input(self):
        mask = seg_utils.compute_cranium_mask(self._cta(np.float32))
        self.assertEqual(mask.shape, (32, 32, 20))
        self.assertGreater(int(np.count_nonzero(mask)), 0)

    def test_integer_input_is_filtered_in_floating_point(self):
        mask = seg_utils.compute_cranium_mask(self._cta(np.int32))
        self.assertEqual(mask.shape, (32, 32, 20))
        self.assertGreater(int(np.count_nonzero(mask)), 0)


class TestJoinHeadAndNeck(ArterialTestCase):

    def test_head_and_neck_are_stitched_at_the_best_matching_slice(self):
        cta = np.zeros((8, 8, 60), dtype=np.int16)
        cta_affine = np.eye(4)
        head_affine = np.eye(4); head_affine[:3, 3] = [1, 1, 30]
        head = np.zeros((6, 6, 25), dtype=np.uint8)
        neck = np.zeros((8, 8, 33), dtype=np.uint8)
        # a vessel crossing the overlap, seen by both slabs (global voxels 3:5, 3:5, 30:33)
        head[2:4, 2:4, 0:3] = 1
        neck[3:5, 3:5, 30:33] = 1
        head[2:4, 2:4, 10] = 1   # head-only structure at global z = 40
        neck[3:5, 3:5, 5] = 1    # neck-only structure at global z = 5
        joined_nifti, joined = seg_utils.join_head_and_neck_segmentations(cta, cta_affine, head, neck, head_affine)
        self.assertEqual(joined.shape, cta.shape)
        self.assertEqual(joined_nifti.shape, cta.shape)
        self.assertTrue(np.all(joined[3:5, 3:5, 40] == 1), "head structure must land above the seam")
        self.assertTrue(np.all(joined[3:5, 3:5, 5] == 1), "neck structure must land below the seam")
        self.assertEqual(int(joined.sum()), 4 * 3 + 4 + 4)

    def test_non_overlapping_slabs_raise(self):
        cta = np.zeros((8, 8, 60), dtype=np.int16)
        head_affine = np.eye(4); head_affine[:3, 3] = [0, 0, 40]
        with self.assertRaises(ValueError):
            seg_utils.join_head_and_neck_segmentations(cta, np.eye(4), np.zeros((8, 8, 20), np.uint8), np.zeros((8, 8, 33), np.uint8), head_affine)


class TestSliceCta(ArterialTestCase):

    @slow
    def test_laplacian_slicing_of_the_fixture(self):
        cta = nib.load(self.require_fixture("cta.nii.gz"))
        array = np.asarray(cta.dataobj)
        head, neck, head_affine, bbox = seg_utils.slice_cta_head_and_neck(array, cta.affine, use_laplacian=True, return_bounding_box=True)
        self.assertEqual(head.ndim, 3)
        self.assertEqual(neck.shape[:2], array.shape[:2])
        self.assertLess(neck.shape[2], array.shape[2])
        self.assertEqual(head_affine.shape, (4, 4))
        self.assertEqual(len(bbox), 6)
