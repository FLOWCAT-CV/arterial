#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
Unit tests for the pure geometry and feature helpers of feature_extraction,
on synthetic curves and graphs with known answers.
"""

import math
import unittest

import numpy as np
import networkx as nx

from arterial.feature_extraction import utils as fe_utils
from arterial.feature_extraction.local_features import utils as local_utils
from arterial.feature_extraction.segment_features import utils as segment_utils


def segment_graph(positions, radii=None, blanking=None, curvature=None):
    """
    Builds a path-shaped segment with the node attributes the feature helpers read.

    """
    n = len(positions)
    radii = radii if radii is not None else [1.0] * n
    blanking = blanking if blanking is not None else [0] * n
    curvature = curvature if curvature is not None else [0.0] * n
    graph = nx.path_graph(n)
    for idx in range(n):
        graph.nodes[idx]["pos"] = np.asarray(positions[idx], dtype=float)
        graph.nodes[idx]["hierarchy femoral"] = idx
        graph.nodes[idx]["features femoral"] = {"blanking": blanking[idx], "radius": radii[idx], "curvature": curvature[idx]}
    return graph


class TestSphericalAngles(unittest.TestCase):

    def test_axis_vectors(self):
        module, polar, azimuth = local_utils.compute_spherical_angles([0, 0, 2])
        self.assertAlmostEqual(module, 2)
        self.assertAlmostEqual(polar, math.pi / 2)
        self.assertTrue(math.isnan(azimuth), "azimuth is undefined along z")
        module, polar, azimuth = local_utils.compute_spherical_angles([3, 0, 0])
        self.assertAlmostEqual(module, 3)
        self.assertAlmostEqual(polar, 0)
        self.assertAlmostEqual(azimuth, 0)

    def test_diagonal_and_zero(self):
        _, _, azimuth = local_utils.compute_spherical_angles([1, 1, 0])
        self.assertAlmostEqual(azimuth, math.pi / 4)
        _, _, azimuth = local_utils.compute_spherical_angles([1, -1, 0])
        self.assertAlmostEqual(azimuth, -math.pi / 4)
        module, polar, azimuth = local_utils.compute_spherical_angles([0, 0, 0])
        self.assertEqual(module, 0)
        self.assertTrue(math.isnan(polar) and math.isnan(azimuth))


class TestCurvatureAndTorsion(unittest.TestCase):

    def test_circle_has_curvature_one_over_radius_and_no_torsion(self):
        radius = 5.0
        t = np.linspace(0, math.pi / 2, 50)
        curve = np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros_like(t)], axis=1)
        curvature, torsion = local_utils.compute_curvature_and_torsion(curve, 25)
        self.assertAlmostEqual(curvature, 1 / radius, places=3)
        self.assertAlmostEqual(torsion, 0, places=6)

    def test_helix_has_constant_curvature_and_torsion(self):
        r, c = 4.0, 1.5
        t = np.linspace(0, 4 * math.pi, 400)
        curve = np.stack([r * np.cos(t), r * np.sin(t), c * t], axis=1)
        curvature, torsion = local_utils.compute_curvature_and_torsion(curve, 200)
        self.assertAlmostEqual(curvature, r / (r ** 2 + c ** 2), delta=0.01)
        self.assertAlmostEqual(torsion, c / (r ** 2 + c ** 2), delta=0.01)

    def test_straight_line_has_zero_curvature(self):
        curve = np.stack([np.linspace(0, 10, 20), np.zeros(20), np.zeros(20)], axis=1)
        with np.errstate(all="ignore"):
            curvature, _ = local_utils.compute_curvature_and_torsion(curve, 10)
        self.assertAlmostEqual(curvature, 0)


class TestResampling(unittest.TestCase):

    def test_straight_segment_keeps_endpoints_and_spacing(self):
        coords = np.stack([np.arange(10.0), np.zeros(10), np.zeros(10)], axis=1)
        radii = np.arange(10.0)
        new_coords, new_radii = fe_utils.resample_single_segment(coords, radii, target_distance=2.0)
        self.assertEqual(len(new_coords), 6)
        np.testing.assert_allclose(new_coords[0], coords[0], atol=1e-9)
        np.testing.assert_allclose(new_coords[-1], coords[-1], atol=1e-9)
        spacing = np.linalg.norm(np.diff(new_coords, axis=0), axis=1)
        np.testing.assert_allclose(spacing, 1.8, atol=1e-6)
        self.assertEqual(len(new_radii), 6)
        self.assertAlmostEqual(new_radii[0], 0.0)
        self.assertAlmostEqual(new_radii[-1], 9.0)

    def test_short_or_degenerate_segments_are_returned_unchanged(self):
        coords = np.array([[0.0, 0, 0], [0.5, 0, 0]])
        new_coords, new_radii = fe_utils.resample_single_segment(coords, np.array([1.0, 1.0]), target_distance=2.0)
        np.testing.assert_array_equal(new_coords, coords)
        single = np.array([[1.0, 2.0, 3.0]])
        new_coords, _ = fe_utils.resample_single_segment(single, np.array([1.0]), target_distance=2.0)
        np.testing.assert_array_equal(new_coords, single)

    def test_segments_array_resampling_preserves_layout(self):
        array = np.empty((2, 2), dtype=object)
        array[0, 0] = np.stack([np.arange(8.0), np.zeros(8), np.zeros(8)], axis=1); array[0, 1] = np.ones(8)
        array[1, 0] = np.stack([np.zeros(5), np.arange(5.0) * 3, np.zeros(5)], axis=1); array[1, 1] = np.full(5, 2.0)
        resampled = fe_utils.resample_centerline_segments_array(array, 2.0)
        self.assertEqual(resampled.shape, (2, 2))
        for idx in range(2):
            np.testing.assert_allclose(resampled[idx, 0][0], array[idx, 0][0], atol=1e-9)
            np.testing.assert_allclose(resampled[idx, 0][-1], array[idx, 0][-1], atol=1e-9)
            self.assertEqual(len(resampled[idx, 0]), len(resampled[idx, 1]))


class TestGraphHelpers(unittest.TestCase):

    def test_gen_color(self):
        colors = fe_utils.gen_color("viridis", 3)
        self.assertEqual(len(colors), 3)
        self.assertTrue(all(c.startswith("#") and len(c) == 7 for c in colors))
        self.assertNotEqual(colors[0], colors[-1])
        self.assertEqual(len(fe_utils.gen_color("bwr", 1)), 1)

    def test_predicted_vessels_dict(self):
        graph = nx.Graph()
        graph.add_edge(0, 1, cell_id=7, vessel_type=2, vessel_type_name="LCCA")
        graph.add_edge(1, 2, cell_id=3, vessel_type=0, vessel_type_name="AA")
        types, names = fe_utils.get_predicted_vessels_dict(graph)
        self.assertEqual(types, {7: 2, 3: 0})
        self.assertEqual(names, {7: "LCCA", 3: "AA"})

    def test_hierarchical_order_on_path_and_fork(self):
        path = nx.path_graph(4)
        fe_utils.get_hierarchical_order(path, "femoral", start_node=0)
        self.assertEqual([path.nodes[n]["hierarchy femoral"] for n in range(4)], [0, 1, 2, 3])
        fork = nx.Graph([(0, 1), (1, 2), (1, 3), (3, 4)])
        fe_utils.get_hierarchical_order(fork, "radial", start_node=0)
        self.assertEqual(fork.nodes[2]["hierarchy radial"], 2)
        self.assertEqual(fork.nodes[3]["hierarchy radial"], 2)
        self.assertEqual(fork.nodes[4]["hierarchy radial"], 3)


class TestSegmentFeatures(unittest.TestCase):

    def test_length_and_tortuosity_of_straight_and_bent_segments(self):
        straight = segment_graph([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)])
        self.assertAlmostEqual(segment_utils.length(straight), 3.0)
        self.assertAlmostEqual(segment_utils.tortuosity_index(straight), 0.0)
        bent = segment_graph([(0, 0, 0), (3, 0, 0), (3, 4, 0)])
        self.assertAlmostEqual(segment_utils.length(bent), 7.0)
        self.assertAlmostEqual(segment_utils.tortuosity_index(bent), 1 - 5 / 7)

    def test_diameters_respect_blanking(self):
        graph = segment_graph([(0, 0, 0), (1, 0, 0), (2, 0, 0)], radii=[1.0, 2.0, 10.0], blanking=[0, 0, 1])
        self.assertAlmostEqual(segment_utils.mean_diameter(graph), 3.0)
        self.assertAlmostEqual(segment_utils.max_diameter(graph), 4.0)
        self.assertAlmostEqual(segment_utils.min_diameter(graph), 2.0)
        all_blanked = segment_graph([(0, 0, 0), (1, 0, 0)], radii=[1.0, 3.0], blanking=[1, 1])
        self.assertAlmostEqual(segment_utils.mean_diameter(all_blanked), 4.0, msg="falls back to every node when all are blanked")

    def test_proximal_and_distal_nodes(self):
        graph = segment_graph([(0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0)], blanking=[1, 0, 0, 1])
        self.assertEqual(segment_utils.find_proximal_node(graph), 1)
        self.assertEqual(segment_utils.find_distal_node(graph), 2)
        self.assertEqual(segment_utils.find_proximal_node(graph, use_blanking=False), 0)
        self.assertEqual(segment_utils.find_distal_node(graph, use_blanking=False), 3)

    def test_bending_length(self):
        straight = segment_graph([(0, 0, 0), (1, 0, 0), (2, 0, 0)])
        self.assertAlmostEqual(segment_utils.bending_length(straight, 0, 2), 0.0)
        bowed = segment_graph([(0, 0, 0), (1, 2, 0), (2, 0, 0)])
        self.assertAlmostEqual(segment_utils.bending_length(bowed, 0, 2), 2.0)

    def test_partial_segment_and_curvature_sums(self):
        positions = [(float(i), 0, 0) for i in range(12)]
        graph = segment_graph(positions, curvature=[0.5] * 12)
        partial = segment_utils.partial_segment(graph, starting_node=0, partial_length_mm=5)
        self.assertGreaterEqual(segment_utils.length(partial), 5.0)
        self.assertLess(partial.number_of_nodes(), 12)
        self.assertAlmostEqual(segment_utils.curvature_energy(graph), 12 * 0.25)
        self.assertAlmostEqual(segment_utils.cumulative_curvature(graph, 0), 11 * 0.5)
