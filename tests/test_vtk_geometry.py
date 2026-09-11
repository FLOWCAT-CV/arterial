#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

"""
vtk_centerline_geometry on synthetic polylines and surfaces with known geometry.
"""

import math
import os

import numpy as np
import networkx as nx
import vtk
from vtk.util.numpy_support import vtk_to_numpy

from helpers import ArterialTestCase
from arterial.io.load_and_save_operations import load_vtkpolydata, save_vtkpolydata
from arterial.feature_extraction.vtk_centerline_geometry import utils as geo
from arterial.feature_extraction.vtk_centerline_geometry import feature_extraction as geo_fe


def polyline(coordinates):
    coordinates = np.asarray(coordinates, dtype=float)
    points = vtk.vtkPoints()
    for point in coordinates:
        points.InsertNextPoint(*point)
    line = vtk.vtkPolyLine(); line.GetPointIds().SetNumberOfIds(len(coordinates))
    for idx in range(len(coordinates)):
        line.GetPointIds().SetId(idx, idx)
    cells = vtk.vtkCellArray(); cells.InsertNextCell(line)
    polydata = vtk.vtkPolyData(); polydata.SetPoints(points); polydata.SetLines(cells)
    return polydata


def with_frenet(polydata):
    coordinates = vtk_to_numpy(polydata.GetPoints().GetData())
    tangents, normals, binormals = geo.compute_frenet_frame(coordinates)
    geo.add_point_array(polydata, tangents, "Tangents")
    geo.add_point_array(polydata, normals, "Normals")
    geo.add_point_array(polydata, binormals, "Binormals")
    return polydata


def cylinder(radius=2.0, height=20.0, resolution=120):
    source = vtk.vtkCylinderSource(); source.SetRadius(radius); source.SetHeight(height); source.SetResolution(resolution); source.CappingOn(); source.Update()
    # vtkCylinderSource's axis is y; rotate so the axis is z
    transform = vtk.vtkTransform(); transform.RotateX(90)
    filt = vtk.vtkTransformPolyDataFilter(); filt.SetTransform(transform); filt.SetInputConnection(source.GetOutputPort()); filt.Update()
    triangles = vtk.vtkTriangleFilter(); triangles.SetInputConnection(filt.GetOutputPort()); triangles.Update()
    return triangles.GetOutput()


class TestPlaneGeometry(ArterialTestCase):

    def test_plane_basis_is_orthonormal(self):
        normal = np.array([1.0, 2.0, 3.0]) / math.sqrt(14)
        u, v = geo.plane_basis(normal)
        for vector in (u, v):
            self.assertAlmostEqual(np.linalg.norm(vector), 1.0)
            self.assertAlmostEqual(float(vector @ normal), 0.0)
        self.assertAlmostEqual(float(u @ v), 0.0)

    def test_projection_preserves_in_plane_distances(self):
        t = np.linspace(0, 2 * math.pi, 40, endpoint=False)
        circle = np.stack([3 * np.cos(t), 3 * np.sin(t), np.full_like(t, 7.0)], axis=1)
        projected = geo.project_to_plane_2d(circle, np.array([0.0, 0.0, 7.0]), np.array([0.0, 0.0, 1.0]))
        self.assertEqual(projected.shape, (40, 2))
        np.testing.assert_allclose(np.linalg.norm(projected, axis=1), 3.0)

    def test_minimum_enclosing_circle(self):
        t = np.linspace(0, 2 * math.pi, 50, endpoint=False)
        cx, cy, r = geo.minimum_enclosing_circle(np.stack([1 + 3 * np.cos(t), -2 + 3 * np.sin(t)], axis=1))
        self.assertAlmostEqual(cx, 1.0, places=6); self.assertAlmostEqual(cy, -2.0, places=6); self.assertAlmostEqual(r, 3.0, places=6)
        cx, cy, r = geo.minimum_enclosing_circle(np.array([[0.0, 0.0], [4.0, 0.0]]))
        self.assertAlmostEqual(cx, 2.0); self.assertAlmostEqual(cy, 0.0); self.assertAlmostEqual(r, 2.0)


class TestFrames(ArterialTestCase):

    def test_frenet_frame_on_straight_line_and_circle(self):
        line = np.stack([np.linspace(0, 10, 11), np.zeros(11), np.zeros(11)], axis=1)
        tangents, normals, binormals = geo.compute_frenet_frame(line)
        np.testing.assert_allclose(tangents, np.tile([1, 0, 0], (11, 1)))
        np.testing.assert_allclose(np.einsum("ij,ij->i", tangents, normals), 0, atol=1e-12)
        np.testing.assert_allclose(binormals, np.cross(tangents, normals))
        t = np.linspace(0, math.pi, 100)
        circle = np.stack([10 * np.cos(t), 10 * np.sin(t), np.zeros_like(t)], axis=1)
        tangents, normals, binormals = geo.compute_frenet_frame(circle)
        for frame in (tangents, normals, binormals):
            np.testing.assert_allclose(np.linalg.norm(frame, axis=1), 1.0, atol=1e-9)
        np.testing.assert_allclose(np.einsum("ij,ij->i", tangents, normals), 0, atol=1e-9)
        # the rotation-minimising frame must not flip between consecutive points
        self.assertTrue(np.all(np.einsum("ij,ij->i", normals[:-1], normals[1:]) > 0.9))

    def test_walk_chain_orders_nodes_and_falls_back_on_cycles(self):
        chain = nx.Graph([(7, 3), (3, 11), (11, 2)])
        ordered = geo.walk_chain(chain)
        self.assertIn(ordered, ([7, 3, 11, 2], [2, 11, 3, 7]))
        cycle = nx.cycle_graph(4)
        self.assertEqual(sorted(geo.walk_chain(cycle)), [0, 1, 2, 3])

    def test_lpi_corner_coordinates(self):
        affine = np.diag([-0.5, -0.5, 0.8, 1.0]); affine[:3, 3] = [100, 120, -30]  # LPS
        corner = geo.lpi_corner_coordinates(affine, (512, 512, 782))
        np.testing.assert_allclose(corner, (affine @ np.array([511, 511, 0, 1]))[:3])
        np.testing.assert_allclose(geo.lpi_corner_coordinates(np.eye(4), (10, 10, 10)), [0, 0, 0])
        with self.assertRaises(ValueError):
            geo.lpi_corner_coordinates(np.diag([1.0, 1.0, -1.0, 1.0]), (10, 10, 10))


class TestArraysAndFixture(ArterialTestCase):

    def test_point_arrays_survive_a_legacy_vtk_roundtrip(self):
        polydata = polyline([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
        geo.add_point_array(polydata, np.array([1.5, 2.5, 3.5]), "Radius")
        geo.add_point_array(polydata, np.eye(3), "Tangents")
        geo.add_string_point_array(polydata, ["AA", "BT", "LCCA"], "VesselTypeName")
        path = os.path.join(self.case_dir, "line.vtk")
        save_vtkpolydata(polydata, path)
        loaded = load_vtkpolydata(path)
        np.testing.assert_allclose(vtk_to_numpy(loaded.GetPointData().GetArray("Radius")), [1.5, 2.5, 3.5])
        self.assertEqual(vtk_to_numpy(loaded.GetPointData().GetArray("Tangents")).shape, (3, 3))
        names = loaded.GetPointData().GetAbstractArray("VesselTypeName")
        self.assertEqual([names.GetValue(i) for i in range(3)], ["AA", "BT", "LCCA"])

    def test_blanking_lookup_on_the_fixture_branch_model(self):
        model = load_vtkpolydata(self.require_fixture("branch_model.vtk"))
        coordinates, blanking = geo.branch_model_blanking_lookup(model)
        expected = sum(model.GetCell(c).GetNumberOfPoints() for c in range(model.GetNumberOfCells()))
        self.assertEqual(len(coordinates), expected)
        self.assertEqual(len(blanking), expected)
        self.assertTrue(set(np.unique(blanking).tolist()) <= {0, 1})
        np.testing.assert_allclose(coordinates[0], model.GetCell(0).GetPoints().GetPoint(0))


class TestCrossSections(ArterialTestCase):

    def test_cross_section_area_of_a_cylinder(self):
        surface = cylinder(radius=2.0)
        section = geo.extract_cross_section(surface, (0.0, 0.0, 0.0), (0.0, 0.0, 1.0))
        self.assertGreater(section.GetNumberOfPoints(), 0)
        self.assertAlmostEqual(geo.cross_section_area(section), math.pi * 4, delta=0.05 * math.pi * 4)
        empty = geo.extract_cross_section(surface, (0.0, 0.0, 50.0), (0.0, 0.0, 1.0))
        self.assertEqual(empty.GetNumberOfPoints(), 0)

    def test_radius_extraction_inside_and_outside_the_surface(self):
        surface = cylinder(radius=2.0)
        centerline = with_frenet(polyline([[0, 0, z] for z in np.linspace(-5, 5, 6)]))
        geo.add_point_array(centerline, np.full(6, 1.7), "MaximumInscribedSphereRadius")
        out = geo_fe.perform_radius_extraction(centerline, surface)
        ce = vtk_to_numpy(out.GetPointData().GetArray("Radius CE")); cc = vtk_to_numpy(out.GetPointData().GetArray("Radius CC"))
        np.testing.assert_allclose(ce, 2.0, rtol=0.05); np.testing.assert_allclose(cc, 2.0, rtol=0.05)
        np.testing.assert_allclose(vtk_to_numpy(out.GetPointData().GetArray("Ovality")), 1.7 / cc)
        far = with_frenet(polyline([[0, 0, z] for z in np.linspace(50, 60, 6)]))  # planes beyond the cylinder's ends
        geo.add_point_array(far, np.full(6, 1.7), "MaximumInscribedSphereRadius")
        out = geo_fe.perform_radius_extraction(far, surface)
        np.testing.assert_allclose(vtk_to_numpy(out.GetPointData().GetArray("Radius CE")), 1.7)
        np.testing.assert_allclose(vtk_to_numpy(out.GetPointData().GetArray("Ovality")), 1.0)

    def test_missing_arrays_raise(self):
        with self.assertRaises(RuntimeError):
            geo_fe.perform_radius_extraction(polyline([[0, 0, 0], [0, 0, 1]]), cylinder())


class TestCurvature(ArterialTestCase):

    def test_curvature_arrays_on_a_half_circle(self):
        radius, n = 10.0, 200
        t = np.linspace(0, math.pi, n)
        centerline = with_frenet(polyline(np.stack([radius * np.cos(t), radius * np.sin(t), np.zeros(n)], axis=1)))
        out = geo_fe.perform_curvature_extraction(centerline, savgol_window_length=21, savgol_polyorder=3)
        distance = vtk_to_numpy(out.GetPointData().GetArray("Distance from origin"))
        self.assertAlmostEqual(distance[-1], math.pi * radius, delta=0.01)
        spacing = distance[1]
        curvature = vtk_to_numpy(out.GetPointData().GetArray("Curvature"))
        # Curvature is per sample (gradient over the index), i.e. spacing / radius, not 1 / radius.
        np.testing.assert_allclose(curvature[10:-10], spacing / radius, rtol=0.02)
        cumulative = vtk_to_numpy(out.GetPointData().GetArray("Cumulative angle of curvature"))
        self.assertAlmostEqual(cumulative[-1], 180.0, delta=2.0)
        for name in ["Torsion", "Filtered curvature", "Angle of curvature"]:
            self.assertEqual(out.GetPointData().GetArray(name).GetNumberOfTuples(), n)

    def test_straight_line_has_no_curvature(self):
        centerline = with_frenet(polyline([[0, 0, z] for z in np.linspace(0, 30, 61)]))
        out = geo_fe.perform_curvature_extraction(centerline, savgol_window_length=11, savgol_polyorder=2)
        np.testing.assert_allclose(vtk_to_numpy(out.GetPointData().GetArray("Curvature")), 0, atol=1e-12)

    def test_curve_ids_split_at_the_minimum_between_peaks(self):
        n = 200
        centerline = polyline([[0, 0, z] for z in range(n)])
        x = np.arange(n)
        filtered = 0.1 * np.exp(-((x - 60) / 8) ** 2) + 0.1 * np.exp(-((x - 140) / 8) ** 2)
        geo.add_point_array(centerline, filtered, "Filtered curvature")
        out = geo_fe.perform_curve_id_extraction(centerline, peak_height=0.03, peak_width=5)
        ids = vtk_to_numpy(out.GetPointData().GetArray("CurveIds"))
        self.assertEqual(ids[0], 0); self.assertEqual(ids[-1], 1)
        self.assertEqual(int(np.argmax(ids)), 100)
        with self.assertRaises(ValueError):
            geo_fe.perform_curve_id_extraction(polyline([[0, 0, 0], [0, 0, 1]]))
