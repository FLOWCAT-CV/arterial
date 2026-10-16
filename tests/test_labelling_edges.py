#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
import tempfile
import unittest

import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import Compose, RadiusGraph

from arterial.vessel_labelling.utils import get_edge_transforms
from arterial.vessel_labelling.models import ResGAT
from arterial.vessel_labelling.inference import load_vessel_labelling_model


def node_form_graph():
    # Four segments: 0, 1, 2 meet at one bifurcation (one direction per edge, as node_transform gives);
    # segment 3 touches nothing but lies 5 mm from segment 2. Positions in mm.
    pos = torch.tensor([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 45.0, 0.0]])
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
    return Data(x=torch.randn(4, 24), pos=pos, edge_index=edge_index, num_nodes=4)


def edge_set(edge_index):
    return set(map(tuple, edge_index.T.tolist()))


class TestEdgeTransforms(unittest.TestCase):

    def apply(self, edge_mode, radius_mm=25.0):
        return Compose(get_edge_transforms(edge_mode, radius_mm))(node_form_graph())

    def test_radius_only_matches_the_original_transform(self):
        original = RadiusGraph(r=0.5, max_num_neighbors=10)(node_form_graph())
        self.assertTrue(torch.equal(self.apply("radius_only").edge_index, original.edge_index))
        # No segments are within 0.5 mm, so every topological edge is dropped
        self.assertEqual(self.apply("radius_only").edge_index.shape[1], 0)

    def test_topological_edges_become_undirected(self):
        edges = edge_set(self.apply("topological").edge_index)
        self.assertEqual(edges, {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)})

    def test_topological_plus_radius_adds_close_segments_without_duplicates(self):
        edge_index = self.apply("topological+radius", radius_mm=10.0).edge_index
        edges = edge_set(edge_index)
        self.assertEqual(edges, {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1), (2, 3), (3, 2)})
        self.assertEqual(len(edges), edge_index.shape[1])
        self.assertFalse(any(a == b for a, b in edges))

    def test_unknown_edge_mode_raises(self):
        with self.assertRaises(ValueError):
            get_edge_transforms("radius")


class TestLoadVesselLabellingModel(unittest.TestCase):

    def save(self, checkpoint):
        path = os.path.join(self.tmp.name, "model_weights.pth")
        torch.save(checkpoint, path)
        return path

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.init_kwargs = dict(in_channels=24, hidden_channels=8, out_channels=14, num_layers=3, act=F.leaky_relu)

    def tearDown(self):
        self.tmp.cleanup()

    def test_checkpoints_without_metadata_load_as_gat_radius_only(self):
        from torch_geometric.nn.models import GAT
        gat = GAT(v2=True, **self.init_kwargs)
        model, edge_config = load_vessel_labelling_model(self.save(dict(state_dict=gat.state_dict(), init_kwargs=self.init_kwargs)), "cpu")
        self.assertIsInstance(model, GAT)
        self.assertEqual(edge_config, {"edge_mode": "radius_only", "radius_mm": 25.0})

    def test_resgat_checkpoint_round_trip(self):
        resgat = ResGAT(**self.init_kwargs).eval()
        path = self.save(dict(state_dict=resgat.state_dict(), init_kwargs=self.init_kwargs, model_class="ResGAT", edge_mode="topological", radius_mm=25.0))
        model, edge_config = load_vessel_labelling_model(path, "cpu")
        self.assertEqual(edge_config["edge_mode"], "topological")
        graph = Compose(get_edge_transforms("topological"))(node_form_graph())
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(graph.x, graph.edge_index), resgat(graph.x, graph.edge_index)))

    def test_unknown_model_class_raises(self):
        path = self.save(dict(state_dict={}, init_kwargs=self.init_kwargs, model_class="MLP"))
        with self.assertRaises(ValueError):
            load_vessel_labelling_model(path, "cpu")


if __name__ == "__main__":
    unittest.main()
