#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
import tempfile
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.transforms import Compose, RadiusGraph

from arterial.vessel_labelling.utils import find_root, get_edge_transforms
from arterial.vessel_labelling.models import ResGAT
from arterial.vessel_labelling.inference import load_vessel_labelling_model, perform_inference


def node_form_graph():
    # Four segments: 0, 1, 2 meet at one bifurcation (one direction per edge, as node_transform gives);
    # segment 3 touches nothing. Positions in mm.
    pos = torch.tensor([[0.0, 0.0, 0.0], [30.0, 0.0, 0.0], [0.0, 40.0, 0.0], [0.0, 45.0, 0.0]])
    edge_index = torch.tensor([[0, 0, 1], [1, 2, 2]])
    return Data(x=torch.randn(4, 24), pos=pos, edge_index=edge_index, num_nodes=4)


def edge_set(edge_index):
    return set(map(tuple, edge_index.T.tolist()))


class TestEdgeTransforms(unittest.TestCase):

    def apply(self, edge_mode):
        return Compose(get_edge_transforms(edge_mode))(node_form_graph())

    def test_radius_only_matches_the_original_transform(self):
        original = RadiusGraph(r=0.5, max_num_neighbors=10)(node_form_graph())
        self.assertTrue(torch.equal(self.apply("radius_only").edge_index, original.edge_index))
        # No segments are within 0.5 mm, so every topological edge is dropped
        self.assertEqual(self.apply("radius_only").edge_index.shape[1], 0)

    def test_topological_edges_become_undirected(self):
        edges = edge_set(self.apply("topological").edge_index)
        self.assertEqual(edges, {(0, 1), (1, 0), (0, 2), (2, 0), (1, 2), (2, 1)})

    def test_unknown_edge_mode_raises(self):
        with self.assertRaises(ValueError):
            get_edge_transforms("topological+radius")


class TestSingleSegmentGraph(unittest.TestCase):

    def test_graph_without_topological_edges_gets_an_empty_edge_index(self):
        import networkx as nx
        import numpy as np
        from arterial import model_registry
        from arterial.vessel_labelling.utils import EVCDatasetInference
        if not os.path.exists(model_registry.model_path("vessel_labelling", "extracranial_vessels", "dataset.json")):
            self.skipTest("vessel labelling dataset.json missing (download the model weights)")
        graph = nx.Graph()
        graph.add_edge(0, 1, cell_id=7, pos=np.zeros(3), features=np.ones(24))
        data = EVCDatasetInference(raw_graph=graph, pre_transform=Compose(get_edge_transforms("topological"))).data_list[0]
        self.assertEqual(tuple(data.edge_index.shape), (2, 0))
        self.assertEqual(data.num_nodes, 1)


class TestRootCentredPositions(unittest.TestCase):

    def graph(self):
        import networkx as nx
        import numpy as np
        # Aortic arch root (widest endpoint) at (10, 20, 30) mm, a bifurcation and two thinner endpoints
        graph = nx.Graph()
        for node, pos, radius in [(0, (10, 20, 30), 12.0), (1, (10, 20, 80), 8.0), (2, (0, 20, 150), 3.0), (3, (40, 20, 150), 4.0)]:
            graph.add_node(node, pos=np.array(pos, float), radius=radius)
        for cell_id, (n0, n1) in enumerate([(0, 1), (1, 2), (1, 3)]):
            features = np.arange(24, dtype=float)
            features[15:24] = np.concatenate([graph.nodes[n0]["pos"], graph.nodes[n1]["pos"], (graph.nodes[n0]["pos"] + graph.nodes[n1]["pos"]) / 2])
            graph.add_edge(n0, n1, cell_id=cell_id, pos=features[21:24], features=features)
        return graph

    def test_find_root_is_the_widest_endpoint(self):
        self.assertEqual(find_root(self.graph()).tolist(), [10.0, 20.0, 30.0])

    def test_position_features_are_relative_to_the_root(self):
        from arterial import model_registry
        from arterial.vessel_labelling.utils import EVCDatasetInference
        if not os.path.exists(model_registry.model_path("vessel_labelling", "extracranial_vessels", "dataset.json")):
            self.skipTest("vessel labelling dataset.json missing (download the model weights)")
        graph = self.graph()
        normalised = EVCDatasetInference(raw_graph=graph).data_list[0].x
        centred = EVCDatasetInference(raw_graph=graph, position_mode="root_centred").data_list[0].x
        raw = torch.tensor(np.array([graph[u][v]["features"][15:24] for u, v in graph.edges]), dtype=torch.float32)
        self.assertTrue(torch.allclose(centred[:, 15:24], (raw - torch.tensor([10.0, 20.0, 30.0]).repeat(3)) / 100))
        # The other 15 features are normalised as for the legacy weights
        self.assertTrue(torch.equal(centred[:, :15], normalised[:, :15]))

    def test_unknown_position_mode_raises(self):
        from arterial.vessel_labelling.utils import EVCDatasetInference
        with self.assertRaises(ValueError):
            EVCDatasetInference(raw_graph=self.graph(), position_mode="world")


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
        self.assertEqual(edge_config, {"edge_mode": "radius_only", "position_mode": None, "position_scale_mm": 100.0})

    def test_resgat_batch_norm_checkpoint_round_trip(self):
        init_kwargs = dict(self.init_kwargs, act="leaky_relu", norm="batch_norm")
        resgat = ResGAT(**dict(init_kwargs, act=F.leaky_relu))
        # Non-trivial running statistics, so that evaluation mode matters
        resgat.train()(torch.randn(16, 24), torch.empty(2, 0, dtype=torch.long))
        resgat.eval()
        path = self.save(dict(state_dict=resgat.state_dict(), init_kwargs=init_kwargs, model_class="ResGAT", edge_mode="topological",
                              position_mode="root_centred", position_scale_mm=100.0))
        model, edge_config = load_vessel_labelling_model(path, "cpu")
        self.assertFalse(model.training)
        self.assertEqual(edge_config, {"edge_mode": "topological", "position_mode": "root_centred", "position_scale_mm": 100.0})
        graph = Compose(get_edge_transforms("topological"))(node_form_graph())
        with torch.no_grad():
            self.assertTrue(torch.allclose(model(graph.x, graph.edge_index), resgat(graph.x, graph.edge_index)))

    def test_unknown_model_class_raises(self):
        path = self.save(dict(state_dict={}, init_kwargs=self.init_kwargs, model_class="MLP"))
        with self.assertRaises(ValueError):
            load_vessel_labelling_model(path, "cpu")

    def test_unknown_model_version_raises(self):
        import networkx as nx
        with self.assertRaises(ValueError):
            perform_inference(nx.Graph(), model_version="v3")


if __name__ == "__main__":
    unittest.main()
