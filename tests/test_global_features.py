#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import unittest

import numpy as np
import networkx as nx

from arterial.feature_extraction.global_features import utils as global_utils


def arch_graph(bt_z=6.0, lcca_radius=2.0, bovine=False, with_lsa=True, rsa_first=True):
    """
    Builds a small labelled graph: an aortic arch (AA) with apex at z=10 and
    radius 2, a BT origin at height bt_z, an LCCA origin, and RSA/LSA origins.

    """
    graph = nx.Graph()

    def node(idx, name, pos, radius=1.0, hierarchy=0):
        graph.add_node(idx, vessel_type_name=name, pos=np.array(pos, dtype=float), radius=radius,
                       **{"features femoral": {"blanking": 0}, "hierarchy femoral": hierarchy})

    node(0, "AA", (0, 0, 0), 2.0, 0)
    node(1, "AA", (0, 5, 10), 2.0, 1)   # apex: A = 10 + 2 = 12
    node(2, "AA", (0, 10, 0), 2.0, 2)
    node(3, "BT", (0, 4, bt_z), 1.0, 3)
    node(4, "LCCA", (0, 6, 9), lcca_radius, 4)
    node(5, "RSA", (0, 3, 8), 1.0, 2)
    graph.add_edge(0, 1, is_artificial=False, vessel_type_name="AA")
    graph.add_edge(1, 2, is_artificial=False, vessel_type_name="AA")
    graph.add_edge(1, 3, is_artificial=False, vessel_type_name="BT")
    # LCCA leaves the BT (bovine) or a different arch node than the BT (normal)
    graph.add_edge(3 if bovine else 2, 4, is_artificial=False, vessel_type_name="LCCA")
    graph.add_edge(1, 5, is_artificial=False, vessel_type_name="AA" if rsa_first else "RSA")
    if with_lsa:
        node(6, "LSA", (0, 8, 8), 1.0, 5 if rsa_first else 1)
        graph.add_edge(2, 6, is_artificial=False, vessel_type_name="LSA")
    return graph


class TestAorticArchType(unittest.TestCase):

    def test_types_by_ratio(self):
        # ratio = |A - B| / D with A = 12, D = 2 * lcca_radius
        self.assertEqual(global_utils.get_aortic_arch_type(arch_graph(bt_z=11.0, lcca_radius=2.0)), 1)   # 0.25
        self.assertEqual(global_utils.get_aortic_arch_type(arch_graph(bt_z=6.0, lcca_radius=2.0)), 2)    # 1.5
        self.assertEqual(global_utils.get_aortic_arch_type(arch_graph(bt_z=2.0, lcca_radius=2.0)), 3)    # 2.5

    def test_exact_boundaries_do_not_crash(self):
        self.assertEqual(global_utils.get_aortic_arch_type(arch_graph(bt_z=8.0, lcca_radius=2.0)), 2)    # ratio 1.0
        self.assertEqual(global_utils.get_aortic_arch_type(arch_graph(bt_z=4.0, lcca_radius=2.0)), 3)    # ratio 2.0

    def test_missing_bt_defaults_to_type_one(self):
        graph = arch_graph()
        graph.remove_node(3)
        self.assertEqual(global_utils.get_aortic_arch_type(graph), 1)

    def test_aortic_arch_coordinates(self):
        coords = global_utils.get_aortic_arch_node_coordinates(arch_graph())
        self.assertEqual(coords.shape, (3, 3))


class TestBovineAndArsa(unittest.TestCase):

    def test_bovine_arch_detected_when_lcca_leaves_the_bt(self):
        self.assertEqual(global_utils.get_bovine_arch(arch_graph(bovine=False)), 0)
        self.assertEqual(global_utils.get_bovine_arch(arch_graph(bovine=True)), 1)

    def test_arsa(self):
        self.assertEqual(global_utils.get_arsa(arch_graph(rsa_first=True)), 1)
        self.assertEqual(global_utils.get_arsa(arch_graph(rsa_first=False)), 0)

    def test_arsa_without_lsa_does_not_crash(self):
        self.assertEqual(global_utils.get_arsa(arch_graph(with_lsa=False)), 0)
