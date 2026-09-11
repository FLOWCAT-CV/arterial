#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import unittest

import numpy as np
import nibabel as nib
import torch
from torch_geometric.data import Data

from arterial.access_prediction import utils as ap_utils
from arterial.access_prediction.models import ArterialGNet
from arterial.access_prediction.preprocessing.utils import get_onehot_encoded_vessel_type


class TestLpiCorner(unittest.TestCase):

    def nifti(self, diag, shape=(10, 12, 14), translation=(5.0, -7.0, 3.0)):
        affine = np.diag(list(diag) + [1.0]); affine[:3, 3] = translation
        return nib.Nifti1Image(np.zeros(shape, dtype=np.int16), affine)

    def test_supported_orientations(self):
        np.testing.assert_allclose(ap_utils.get_lpi_corner_coordinates(self.nifti((1, 1, 1))), [5, -7, 3])
        np.testing.assert_allclose(ap_utils.get_lpi_corner_coordinates(self.nifti((-1, 1, 1))), [5 - 9, -7, 3])
        np.testing.assert_allclose(ap_utils.get_lpi_corner_coordinates(self.nifti((-1, -1, 1))), [5 - 9, -7 - 11, 3])

    def test_unsupported_orientation_raises(self):
        with self.assertRaises(ValueError):
            ap_utils.get_lpi_corner_coordinates(self.nifti((1, 1, -1)))


class TestOneHot(unittest.TestCase):

    def test_single_and_multiple_vessel_types(self):
        single = get_onehot_encoded_vessel_type("AA")
        self.assertEqual(single.shape, (14,))
        self.assertEqual(single.sum(), 1)
        multiple = get_onehot_encoded_vessel_type(["AA", "BT"])
        self.assertEqual(multiple.sum(), 2)


def synthetic_item(global_dim=5, segment_dim=6, edge_dim=3, dense_dim=7):
    item = Data()
    item.global_data = torch.randn(global_dim)
    segment = Data(x=torch.randn(4, segment_dim), edge_index=torch.tensor([[0, 1, 2], [1, 2, 3]]), edge_attr=torch.randn(3, edge_dim),
                   pos=torch.rand(4, 3) * 50)
    segment.num_nodes = 4
    dense = Data(x=torch.randn(12, dense_dim), edge_index=torch.stack([torch.arange(11), torch.arange(1, 12)]), pos=torch.rand(12, 3) * 20)
    dense.num_nodes = 12
    item.segment_data = segment
    item.dense_data = dense
    return item


class TestBatching(unittest.TestCase):

    def test_radius_graph_adds_edges_without_duplicates(self):
        item = ap_utils.DenseRadiusGraph(r=10)(synthetic_item())
        edges = item.dense_data.edge_index
        self.assertGreaterEqual(edges.shape[1], 11)
        self.assertEqual(torch.unique(edges, dim=1).shape[1], edges.shape[1])

    def test_collate_stacks_globals_and_batches_graphs(self):
        batch = ap_utils.collate_ArterialGNetInference([synthetic_item(), synthetic_item()])
        self.assertEqual(tuple(batch.global_data.shape), (2, 5))
        self.assertEqual(batch.dense_data.num_nodes, 24)
        self.assertEqual(int(batch.batch.max()), 1)
        self.assertIs(batch.batch, batch.dense_data.batch)


class TestArterialGNet(unittest.TestCase):

    def build(self, **overrides):
        kwargs = dict(global_in_dim=5, segment_node_in_dim=6, segment_edge_in_dim=3, dense_node_in_dim=7, hidden_dim=8, hidden_dim_dense=8,
                      out_dim=2, num_global_layers=1, num_segment_layers=1, num_dense_layers=1, num_out_layers=2, attn_heads=4,
                      aggregation="mean", dropout=0.0, concat=False, is_classification=True)
        kwargs.update(overrides)
        return ArterialGNet(**kwargs)

    def test_forward_returns_probabilities_and_attention(self):
        model = self.build().eval()
        batch = ap_utils.collate_ArterialGNetInference([ap_utils.DenseRadiusGraph(r=10)(synthetic_item())])
        with torch.no_grad():
            out, (edge_index, weights) = model(batch)
        self.assertEqual(tuple(out.shape), (1, 2))
        self.assertAlmostEqual(float(out.sum()), 1.0, places=5)
        self.assertEqual(edge_index.shape[0], 2)
        self.assertEqual(weights.shape, (edge_index.shape[1], 4), "one attention weight per edge and head")

    def test_unknown_aggregation_is_rejected(self):
        with self.assertRaises(ValueError):
            self.build(aggregation="sum")


class TestAttentionMap(unittest.TestCase):

    def test_attention_graph_has_positions_offset_by_the_corner(self):
        item = synthetic_item()
        batch = ap_utils.collate_ArterialGNetInference([item])
        edge_index = batch.dense_data.edge_index.numpy()
        weights = np.random.rand(edge_index.shape[1], 4)
        graph = ap_utils.build_final_attention_map(edge_index, weights, batch, np.array([100.0, 200.0, 300.0]))
        self.assertEqual(graph.number_of_nodes(), 12)
        self.assertEqual(graph.number_of_edges(), 11)
        for node, data in graph.nodes(data=True):
            self.assertGreaterEqual(data["pos"][0], 100.0)
        polydata = ap_utils.nx_graph_to_vtk_polydata(graph)
        self.assertEqual(polydata.GetNumberOfPoints(), 12)
        points = ap_utils.nx_graph_to_point_dict(graph)
        self.assertEqual(len(points["attention_weight"]) if "attention_weight" in points else len(next(iter(points.values()))), 12)
