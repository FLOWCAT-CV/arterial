#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import unittest

import numpy as np
import networkx as nx

from arterial.vessel_labelling.utils import node_transform
from arterial.vessel_labelling.preprocessing.preprocessing import build_nx_graph_from_segments_array
from arterial.vessel_labelling.preprocessing.utils import extract_features_for_labelling


def straight(start, end, n=11):
    return np.linspace(np.asarray(start, float), np.asarray(end, float), n)


class TestNodeTransform(unittest.TestCase):

    def test_edges_become_nodes_and_shared_endpoints_become_edges(self):
        graph = nx.Graph()
        graph.add_edge(0, 1, cell_id=10)
        graph.add_edge(1, 2, cell_id=11)
        graph.add_edge(1, 3, cell_id=12)
        transformed = node_transform(graph)
        self.assertEqual(transformed.number_of_nodes(), 3)
        self.assertEqual(sorted(tuple(sorted(e)) for e in transformed.edges), [(0, 1), (0, 2), (1, 2)])
        self.assertEqual(sorted(transformed.nodes[n]["cell_id"] for n in transformed), [10, 11, 12])


class TestGraphFromSegments(unittest.TestCase):

    def segments(self):
        array = np.empty((2, 2), dtype=object)
        array[0, 0] = straight((0, 0, 0), (10, 0, 0)); array[0, 1] = np.ones(11)
        array[1, 0] = straight((10, 0, 0), (10, 10, 0)); array[1, 1] = np.linspace(1.0, 0.5, 11)
        return array

    def test_touching_segments_share_a_node_and_carry_features(self):
        graph = build_nx_graph_from_segments_array(self.segments())
        self.assertEqual(graph.number_of_nodes(), 3)
        self.assertEqual(graph.number_of_edges(), 2)
        for _, _, data in graph.edges(data=True):
            self.assertEqual(len(data["features"]), 24)
            self.assertTrue(np.all(np.isfinite(data["features"])))
            self.assertIn("direction r", data["features_dict"])
        lengths = sorted(d["features_dict"]["distance"] for _, _, d in graph.edges(data=True))
        np.testing.assert_allclose(lengths, [10, 10])

    def test_closed_or_degenerate_segments_produce_finite_features(self):
        graph = nx.Graph()
        loop = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]], dtype=float)
        graph.add_node(0, pos=loop[0], radius=1.0); graph.add_node(1, pos=loop[-1], radius=0.0)
        graph.add_edge(0, 1, cell_id=0, coordinate_array=loop, radius_array=np.array([1.0, 1.0, 0.5, 0.0]))
        with np.errstate(all="raise"):
            graph = extract_features_for_labelling(graph)
        features = graph[0][1]["features"]
        self.assertTrue(np.all(np.isfinite(features)), f"non-finite features: {graph[0][1]['features_dict']}")
        self.assertEqual(graph[0][1]["features_dict"]["direction r"], 0.0)
        self.assertEqual(graph[0][1]["features_dict"]["proximal/distal radius ratio"], 0.0)
