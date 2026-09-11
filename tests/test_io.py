#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os

import numpy as np
import networkx as nx
import nibabel as nib
import vtk

from helpers import ArterialTestCase
from arterial.io import load_and_save_operations as io_ops


def polyline(points, array_name="Radius"):
    """
    Builds a single-cell vtkPolyData polyline with one point-data array.

    """
    vtk_points = vtk.vtkPoints()
    for point in points:
        vtk_points.InsertNextPoint(*point)
    line = vtk.vtkPolyLine()
    line.GetPointIds().SetNumberOfIds(len(points))
    for idx in range(len(points)):
        line.GetPointIds().SetId(idx, idx)
    cells = vtk.vtkCellArray()
    cells.InsertNextCell(line)
    polydata = vtk.vtkPolyData()
    polydata.SetPoints(vtk_points)
    polydata.SetLines(cells)
    radius = vtk.vtkFloatArray()
    radius.SetName(array_name)
    for idx in range(len(points)):
        radius.InsertNextValue(float(idx))
    polydata.GetPointData().AddArray(radius)
    return polydata


class TestPickleJsonNumpy(ArterialTestCase):

    def test_pickle_roundtrip_preserves_graph(self):
        graph = nx.path_graph(4)
        nx.set_node_attributes(graph, {n: {"pos": np.array([n, 0.0, 0.0])} for n in graph})
        path = os.path.join(self.case_dir, "g.pickle")
        io_ops.save_pickle(graph, path)
        loaded = io_ops.load_pickle(path)
        self.assertEqual(sorted(loaded.edges), sorted(graph.edges))
        np.testing.assert_array_equal(loaded.nodes[3]["pos"], [3.0, 0.0, 0.0])

    def test_json_roundtrip_accepts_numpy_scalars_and_arrays(self):
        path = os.path.join(self.case_dir, "features.json")
        io_ops.save_json({"mean": np.float32(0.5), "count": np.int64(3), "vec": np.arange(3)}, path)
        loaded = io_ops.load_json(path)
        self.assertEqual(loaded, {"mean": 0.5, "count": 3, "vec": [0, 1, 2]})

    def test_json_rejects_unserializable_objects(self):
        with self.assertRaises(TypeError):
            io_ops.save_json({"graph": nx.Graph()}, os.path.join(self.case_dir, "bad.json"))

    def test_numpy_roundtrip_including_object_arrays(self):
        path = os.path.join(self.case_dir, "segments.npy")
        array = np.empty((2, 2), dtype=object)
        array[0, 0] = np.zeros((3, 3)); array[0, 1] = np.ones(3); array[1, 0] = np.zeros((1, 3)); array[1, 1] = np.ones(1)
        io_ops.save_numpy(array, path)
        loaded = io_ops.load_numpy(path)
        self.assertEqual(loaded.shape, (2, 2))
        np.testing.assert_array_equal(loaded[0, 1], [1, 1, 1])

    def test_load_fixtures(self):
        array = io_ops.load_numpy(self.require_fixture("centerline_segments_array.npy"))
        self.assertEqual(array.ndim, 2)
        self.assertEqual(array.shape[1], 2)
        graph = io_ops.load_pickle(self.require_fixture("segments_graph_pred.pickle"))
        self.assertGreater(graph.number_of_edges(), 0)
        for _, _, data in graph.edges(data=True):
            self.assertIn("cell_id", data)


class TestNifti(ArterialTestCase):

    def test_nifti_roundtrip_preserves_affine_and_dtype(self):
        affine = np.diag([0.5, 0.5, 0.8, 1.0]); affine[:3, 3] = [-10, -20, 5]
        image = nib.Nifti1Image(np.arange(60, dtype=np.int16).reshape(3, 4, 5), affine)
        path = os.path.join(self.case_dir, "img.nii.gz")
        io_ops.save_nifti(image, path)
        loaded = io_ops.load_nifti(path)
        np.testing.assert_allclose(loaded.affine, affine, rtol=1e-6)  # NIfTI stores the affine as float32
        self.assertEqual(loaded.get_data_dtype(), np.int16)
        np.testing.assert_array_equal(np.asarray(loaded.dataobj), np.asarray(image.dataobj))

    def test_load_fixture_cta_header(self):
        cta = io_ops.load_nifti(self.require_fixture("cta.nii.gz"))
        self.assertEqual(len(cta.shape), 3)
        self.assertEqual(cta.affine.shape, (4, 4))


class TestVtk(ArterialTestCase):

    def test_vtk_roundtrip_preserves_points_and_arrays(self):
        polydata = polyline([(0, 0, 0), (1, 0, 0), (2, 1, 0)])
        path = os.path.join(self.case_dir, "line.vtk")
        io_ops.save_vtkpolydata(polydata, path)
        loaded = io_ops.load_vtkpolydata(path)
        self.assertEqual(loaded.GetNumberOfPoints(), 3)
        self.assertEqual(loaded.GetNumberOfLines(), 1)
        np.testing.assert_allclose(loaded.GetPoint(2), (2, 1, 0))
        self.assertIsNotNone(loaded.GetPointData().GetArray("Radius"))
        self.assertEqual(loaded.GetPointData().GetArray("Radius").GetValue(2), 2.0)

    def test_load_missing_vtk_raises(self):
        with self.assertRaises(FileNotFoundError):
            io_ops.load_vtkpolydata(os.path.join(self.case_dir, "missing.vtk"))

    def test_save_to_unwritable_path_raises(self):
        with self.assertRaises(IOError):
            io_ops.save_vtkpolydata(polyline([(0, 0, 0), (1, 0, 0)]), os.path.join(self.case_dir, "no_such_dir", "x.vtk"))

    def test_stl_writer_produces_a_binary_stl(self):
        cube = vtk.vtkCubeSource(); cube.Update()
        triangles = vtk.vtkTriangleFilter(); triangles.SetInputConnection(cube.GetOutputPort()); triangles.Update()
        path = os.path.join(self.case_dir, "cube.stl")
        io_ops.save_vtkpolydata_as_stl(triangles.GetOutput(), path)
        self.assertGreater(os.path.getsize(path), 84)
        with open(path, "rb") as handle:
            header = handle.read(84)
        self.assertEqual(int.from_bytes(header[80:84], "little"), 12, "a cube has 12 triangles")

    def test_serialize_roundtrip(self):
        polydata = polyline([(0, 0, 0), (0, 1, 0), (0, 2, 0), (0, 3, 1)])
        restored = io_ops.deserialize_vtk_polydata(io_ops.serialize_vtk_polydata(polydata))
        self.assertEqual(restored.GetNumberOfPoints(), 4)
        np.testing.assert_allclose(restored.GetPoint(3), (0, 3, 1))
        self.assertIsNotNone(restored.GetPointData().GetArray("Radius"))

    def test_load_vtk_list_from_dir_uses_natural_order(self):
        for idx in [0, 1, 2, 10, 11]:
            io_ops.save_vtkpolydata(polyline([(idx, 0, 0), (idx, 1, 0)]), os.path.join(self.case_dir, f"centerlines_{idx}.vtk"))
        with open(os.path.join(self.case_dir, "notes.txt"), "w") as handle:
            handle.write("decoy")
        models = io_ops.load_vtk_list_from_dir(self.case_dir)
        self.assertEqual([model.GetPoint(0)[0] for model in models], [0, 1, 2, 10, 11])

    def test_natural_sort_key(self):
        names = ["m_10.vtk", "m_2.vtk", "m_1.vtk", "M_3.vtk"]
        self.assertEqual(sorted(names, key=io_ops.natural_sort_key), ["m_1.vtk", "m_2.vtk", "M_3.vtk", "m_10.vtk"])

    def test_fixture_branch_model_roundtrip(self):
        model = io_ops.load_vtkpolydata(self.require_fixture("branch_model.vtk"))
        self.assertGreater(model.GetNumberOfPoints(), 0)
        names = {model.GetCellData().GetArrayName(i) for i in range(model.GetCellData().GetNumberOfArrays())}
        self.assertIn("Blanking", names)
        path = os.path.join(self.case_dir, "copy.vtk")
        io_ops.save_vtkpolydata(model, path)
        self.assertEqual(io_ops.load_vtkpolydata(path).GetNumberOfPoints(), model.GetNumberOfPoints())
