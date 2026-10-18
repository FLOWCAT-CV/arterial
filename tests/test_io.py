#    Copyright 2022-2026 Vall d'Hebron Research Institute (VHIR) and Universitat de Barcelona (UB), Barcelona, Spain.
#    SPDX-License-Identifier: PolyForm-Noncommercial-1.0.0

import os
import sys
from unittest import mock

import numpy as np
import networkx as nx
import nibabel as nib
import vtk

from helpers import ArterialTestCase
from arterial.io import load_and_save_operations as io_ops
from arterial.io import dicom_and_nifti, registration


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

    def test_failed_json_write_keeps_the_previous_file_and_no_temporary_file(self):
        path = os.path.join(self.case_dir, "features.json")
        io_ops.save_json({"version": 1}, path)
        with self.assertRaises(TypeError):
            io_ops.save_json({"version": 2, "graph": nx.Graph()}, path)
        self.assertEqual(io_ops.load_json(path), {"version": 1})
        self.assertEqual(os.listdir(self.case_dir), ["features.json"])

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

    def test_failed_nifti_write_keeps_the_previous_file_and_no_temporary_file(self):
        path = os.path.join(self.case_dir, "img.nii.gz")
        io_ops.save_nifti(nib.Nifti1Image(np.zeros((2, 2, 2), np.int16), np.eye(4)), path)
        def failing_save(image, filename):
            open(filename, "wb").close()  # a partial file, then the failure
            raise OSError("disk full")
        with mock.patch.object(io_ops.nib, "save", side_effect=failing_save), self.assertRaises(OSError):
            io_ops.save_nifti(nib.Nifti1Image(np.ones((2, 2, 2), np.int16), np.eye(4)), path)
        self.assertEqual(os.listdir(self.case_dir), ["img.nii.gz"])
        np.testing.assert_array_equal(np.asarray(io_ops.load_nifti(path).dataobj), 0)

    def test_two_file_formats_are_saved_whole(self):
        path = os.path.join(self.case_dir, "img.img")
        io_ops.save_nifti(nib.Nifti1Pair(np.arange(8, dtype=np.int16).reshape(2, 2, 2), np.eye(4)), path)
        self.assertEqual(sorted(os.listdir(self.case_dir)), ["img.hdr", "img.img"])
        np.testing.assert_array_equal(np.asarray(nib.load(path).dataobj), np.arange(8).reshape(2, 2, 2))

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


def oriented_image(axcodes, marker=(1, 2, 3)):
    """
    Builds a 4x5x6 image with one marked voxel, stored in the given orientation.

    """
    data = np.zeros((4, 5, 6), dtype=np.int16)
    data[marker] = 7
    affine = np.diag([0.5, 0.6, 0.7, 1.0]); affine[:3, 3] = [-10, 20, 30]
    image = nib.Nifti1Image(data, affine)
    return image.as_reoriented(nib.orientations.ornt_transform(nib.orientations.io_orientation(affine), nib.orientations.axcodes2ornt(axcodes)))


def marker_world_position(image):
    voxel = np.argwhere(np.asarray(image.dataobj) == 7)[0]
    return nib.affines.apply_affine(image.affine, voxel)


class TestOrientation(ArterialTestCase):

    def test_reorientation_keeps_the_anatomy_in_place(self):
        expected = marker_world_position(oriented_image("RAS"))
        for source in ["RAS", "LAS", "LPS", "RPI"]:
            image = oriented_image(source)
            affine_before = image.affine.copy()
            path = os.path.join(self.case_dir, f"{source}.nii.gz")
            nib.save(image, path)
            for target, converted in [("RAS", registration.convert_to_orientation(image, ("R", "A", "S"))),
                                      ("LPS", registration.convert_to_orientation(image, ("L", "P", "S"))),
                                      ("LAS", dicom_and_nifti.convert_to_las(path)),
                                      ("LPS", dicom_and_nifti.convert_to_lps(path))]:
                with self.subTest(source=source, target=target):
                    self.assertEqual("".join(nib.aff2axcodes(converted.affine)), target)
                    np.testing.assert_allclose(marker_world_position(converted), expected, atol=1e-5)
            np.testing.assert_array_equal(image.affine, affine_before, "the input image must not be modified")


class TestDicomToNifti(ArterialTestCase):

    def fake_dicom2nifti(self, convert_directory):
        module = mock.MagicMock(convert_directory=mock.MagicMock(side_effect=convert_directory))
        return mock.patch.dict(sys.modules, {"dicom2nifti": module, "dicom2nifti.settings": module.settings})

    def test_conversion_errors_are_raised_and_the_temporary_directory_removed(self):
        def broken(input_path, output_dir):
            raise RuntimeError("not a DICOM series")
        output_path = os.path.join(self.case_dir, "cta.nii.gz")
        with self.fake_dicom2nifti(broken), self.assertRaises(RuntimeError):
            dicom_and_nifti.convert_dicom_to_nifti_d2n(self.case_dir, output_path)
        self.assertFalse(os.path.exists(output_path))
        self.assertFalse(os.path.exists(os.path.join(self.case_dir, "temporary_dir_dicom2nifti")))

    def test_a_nifti_left_by_an_earlier_run_is_not_used(self):
        temporary_dir = os.path.join(self.case_dir, "temporary_dir_dicom2nifti")
        os.makedirs(temporary_dir)
        nib.save(nib.Nifti1Image(np.zeros((2, 2, 2), np.int16), np.eye(4)), os.path.join(temporary_dir, "000_stale.nii.gz"))
        def convert(input_path, output_dir):
            nib.save(nib.Nifti1Image(np.ones((2, 2, 2), np.int16), np.eye(4)), os.path.join(output_dir, "series.nii.gz"))
        output_path = os.path.join(self.case_dir, "cta.nii.gz")
        with self.fake_dicom2nifti(convert):
            dicom_and_nifti.convert_dicom_to_nifti_d2n(self.case_dir, output_path)
        np.testing.assert_array_equal(np.asarray(nib.load(output_path).dataobj), 1)
        self.assertFalse(os.path.exists(temporary_dir))
