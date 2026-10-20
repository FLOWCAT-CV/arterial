#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import json
import unittest

import numpy as np
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from arterial.centerline_extraction import utils as cl_utils
from arterial.centerline_extraction.preprocessing import utils as pre_utils
from arterial.centerline_extraction.preprocessing import preprocessing as pre
from arterial.centerline_extraction.postprocessing import postprocessing as post


def segments(*pairs):
    array = np.empty((len(pairs), 2), dtype=object)
    for idx, (coords, radii) in enumerate(pairs):
        array[idx, 0] = np.asarray(coords, dtype=float)
        array[idx, 1] = np.asarray(radii, dtype=float)
    return array


class TestIslands(unittest.TestCase):

    def test_split_keeps_islands_above_threshold_largest_first(self):
        volume = np.zeros((30, 30, 30), dtype=np.uint8)
        volume[2:7, 2:7, 2:7] = 1      # 125 voxels
        volume[15:18, 15:18, 15:18] = 1  # 27 voxels
        volume[25, 25, 25] = 1         # 1 voxel
        affine = np.diag([0.43, 0.43, 0.4, 1.0])  # reference voxel size, so the threshold applies unchanged
        islands = pre_utils.split_segmentation(volume, affine, minimum_island_voxel_size=10)
        self.assertEqual([int(i.sum()) for i in islands], [125, 27])
        self.assertEqual(set(np.unique(islands[0]).tolist()), {0.0, 1.0})
        self.assertFalse(np.any(np.logical_and(islands[0], islands[1])))

    def test_bounding_box(self):
        volume = np.zeros((10, 10, 10), dtype=np.uint8)
        volume[2:5, 3:7, 4:6] = 1
        self.assertEqual(tuple(int(v) for v in pre_utils.get_bounding_box_limits_3d(volume)), (2, 4, 3, 6, 4, 5))


class TestVolumeSanityCheck(unittest.TestCase):

    def test_intracranial_mode(self):
        cl_utils.volume_sanity_check(np.ones((20, 20, 20), dtype=np.uint8), np.eye(4), mode="intracranial_vessels")
        with self.assertRaises(ValueError):
            cl_utils.volume_sanity_check(np.ones((5, 5, 5), dtype=np.uint8), np.eye(4), mode="intracranial_vessels")

    def test_extracranial_mode_rejects_a_tiny_segmentation(self):
        with self.assertRaises(ValueError):
            cl_utils.volume_sanity_check(np.ones((10, 10, 10), dtype=np.uint8), np.eye(4), mode="extracranial_vessels")


class TestVtkImage(unittest.TestCase):

    def test_values_survive_for_any_dtype_and_memory_order(self):
        for dtype in [np.uint8, np.int32, np.float32, np.float64]:
            with self.subTest(dtype=dtype.__name__):
                volume = np.zeros((4, 5, 6), dtype=dtype); volume[1:3, 1:3, 1:3] = 1
                image = pre_utils.numpy_array_to_vtk_image_data(volume)
                self.assertEqual(image.GetDimensions(), (6, 5, 4))
                self.assertAlmostEqual(vtk_to_numpy(image.GetPointData().GetScalars()).sum(), 8.0)
        fortran = np.asfortranarray(np.arange(24, dtype=float).reshape(2, 3, 4))
        image = pre_utils.numpy_array_to_vtk_image_data(fortran)
        np.testing.assert_array_equal(vtk_to_numpy(image.GetPointData().GetScalars()), fortran.ravel(order="C"))


class TestEndpointsJson(unittest.TestCase):

    def test_slicer_markups_structure(self):
        endpoints = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        markups = cl_utils.build_endpoints_json(endpoints, np.eye(4), endpoint_labels=["start", "end"])
        json.dumps(markups)
        fiducial = markups["markups"][0]
        self.assertEqual(fiducial["coordinateSystem"], "RAS")
        self.assertEqual(len(fiducial["controlPoints"]), 2)
        self.assertEqual(fiducial["controlPoints"][0]["label"], "start")
        self.assertEqual(fiducial["controlPoints"][1]["position"], [4.0, 5.0, 6.0])
        self.assertEqual(fiducial["lastUsedControlPointNumber"], 2)

    def test_positions_read_back_as_the_same_ras_points_for_any_orientation(self):
        ras = np.array([[10.0, -20.0, 30.0], [-5.0, 15.0, 40.0]])
        for flips in [(1, 1, 1), (-1, 1, 1), (-1, -1, 1), (1, -1, -1)]:  # RAS, LAS, LPS and RPI voxel grids
            with self.subTest(flips=flips):
                fiducial = cl_utils.build_endpoints_json(ras.tolist(), np.diag(list(flips) + [1.0]))["markups"][0]
                self.assertIn(fiducial["coordinateSystem"], ["RAS", "LPS"], "Slicer only reads RAS or LPS")
                sign = np.array([-1, -1, 1]) if fiducial["coordinateSystem"] == "LPS" else np.ones(3)
                read_back = np.array([point["position"] for point in fiducial["controlPoints"]]) * sign
                np.testing.assert_allclose(read_back, ras)

    def test_wrong_label_count_falls_back_to_generic_labels(self):
        markups = cl_utils.build_endpoints_json([[0, 0, 0]], np.eye(4), endpoint_labels=["a", "b"])
        self.assertEqual(markups["markups"][0]["controlPoints"][0]["label"], "Endpoint-1")


class TestCenterlineSegmentCleaning(unittest.TestCase):

    def test_duplicate_points_and_closed_segments_are_removed(self):
        array = segments(
            ([[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]], [1, 1, 1, 1]),
            ([[5, 5, 5], [6, 5, 5], [5, 5, 5]], [1, 1, 1]),
        )
        cleaned = post.clean_centerlines(array)
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(len(cleaned[0][0]), 3)
        self.assertEqual(len(cleaned[0][1]), 3)

    def test_intracranial_segments_are_removed_by_height_and_radius(self):
        array = segments(
            ([[0, 0, 0], [0, 0, 10]], [3, 3]),      # low, large: kept
            ([[0, 0, 98], [0, 0, 100]], [1, 1]),    # top of the range, small: removed
            ([[0, 0, 96], [0, 0, 100]], [3, 3]),    # top but large: kept
        )
        kept = post.remove_intracranial_arteries(array)
        self.assertEqual(len(kept), 2)
        self.assertEqual(kept[0][1][0], 3)
        self.assertEqual(kept[1][1][0], 3)


class TestEndpointRelocation(unittest.TestCase):

    def test_nearest_component_is_measured_from_the_endpoint_at_the_volume_edge(self):
        segmentation = np.zeros((40, 40, 40), dtype=np.uint8)
        segmentation[23:26, 19:22, 0:3] = 1    # component a few voxels from the endpoint
        segmentation[19:22, 19:22, 7:10] = 1   # component at the centre of the clipped window, farther away
        points = vtk.vtkPoints(); points.InsertNextPoint(20.0, 20.0, 1.0)  # outside the segmentation, near k = 0
        relocated = cl_utils.robust_endpoint_relocation(points, segmentation, np.eye(4), window_size=15)
        np.testing.assert_allclose(relocated.GetPoint(0), [24.0, 20.0, 1.0], atol=1e-6)

    def test_endpoints_inside_the_segmentation_are_kept(self):
        segmentation = np.zeros((10, 10, 10), dtype=np.uint8); segmentation[4:7, 4:7, 4:7] = 1
        relocated = vtk.vtkPoints(); relocated.InsertNextPoint(5.0, 5.0, 5.0); relocated.InsertNextPoint(5.0, 5.0, 6.0)
        used = cl_utils.keep_or_relocate_endpoints([(4.0, 4.0, 4.0), (9.0, 9.0, 9.0)], relocated, segmentation, np.eye(4))
        self.assertEqual(used, [(4.0, 4.0, 4.0), (5.0, 5.0, 6.0)])


class TestFastModeCut(unittest.TestCase):

    def test_top_tenth_of_the_vessels_is_removed(self):
        segmentation = np.zeros((40, 40, 120)); segmentation[17:23, 17:23, 50:100] = 1  # vessel between slices 50 and 99
        model, models = pre.preprocess_segmentation_for_centerline_extraction(segmentation, np.diag([0.43, 0.43, 0.4, 1.0]), True, minimum_island_voxel_size=10)
        bottom, top = model.GetBounds()[4:]
        self.assertAlmostEqual(models[0].GetBounds()[5], bottom + 0.9 * (top - bottom), delta=0.6)


class TestDegreeTwoMerge(unittest.TestCase):

    def test_merged_segment_runs_from_the_free_end_to_the_connected_end(self):
        array = segments(
            ([[0, 0, 0], [0, 0, -1], [0, 0, -2]], [1, 1, 1]),  # free end at z = -2
            ([[0, 0, 0], [0, 0, 1], [0, 0, 2]], [2, 2, 2]),    # end shared with the next segment
            ([[0, 0, 2], [0, 0, 3], [0, 0, 4]], [3, 3, 3]),
        )
        merged = post.merge_degree_2_segments(array)
        self.assertEqual(len(merged), 2)
        np.testing.assert_allclose(merged[0][0][0], [0, 0, -2])
        np.testing.assert_allclose(merged[0][0][-1], [0, 0, 2])
