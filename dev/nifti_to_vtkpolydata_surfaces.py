import nibabel as nib
import vtk
from arterial.io.load_and_save_operations import *

from skimage import measure

segmentation_nifti = load_nifti("/home/vhir/github/arterial/tests/test_data/output/extracranial_vessels/segmentation.nii.gz")
segmentation_array = segmentation_nifti.get_fdata()
segmentation_affine = segmentation_nifti.affine

# Split segmentation_array in a list of arrays with the different islands
def split_segmentation(segmentation_array):
    """
    Splits binary array into a list of binary arrays with the different islands.
    
    """
    label_mask = measure.label(segmentation_array)
    values, counts = np.unique(label_mask, return_counts=True)

    segmentation_array_list = []
    for idx in range(len(values)):
        if counts[idx] >= 1e5:
            segmentation_array_list.append(label_mask == values[idx])

    return segmentation_array_list

segmentation_array_list = split_segmentation(segmentation_array)

import numpy as np
import vtk
from vmtk import vtkvmtk

def numpy_array_to_vtk_image_data(numpy_array):
    # Ensure the numpy array is C-contiguous
    numpy_array = np.ascontiguousarray(numpy_array)
    # Convert a numpy array to a VTK ImageData object
    importer = vtk.vtkImageImport()
    importer.CopyImportVoidPointer(numpy_array, numpy_array.nbytes)
    importer.SetDataScalarTypeToUnsignedChar()
    importer.SetNumberOfScalarComponents(1)
    importer.SetDataExtent(0, numpy_array.shape[2] - 1, 0, numpy_array.shape[1] - 1, 0, numpy_array.shape[0] - 1)
    importer.SetWholeExtent(0, numpy_array.shape[2] - 1, 0, numpy_array.shape[1] - 1, 0, numpy_array.shape[0] - 1)
    importer.Update()
    return importer.GetOutput()

def add_affine_information(vtk_image_data, affine):
    # Set origin and spacing
    origin = affine[:3, 3]
    spacing = np.linalg.norm(affine[:3, :3], axis=0)  # Correct spacing calculation

    vtk_image_data.SetOrigin(origin)
    vtk_image_data.SetSpacing(spacing)

    # Set orientation (direction cosine matrix)
    direction_cosines = vtk.vtkMatrix3x3()
    for i in range(3):
        for j in range(3):
            direction_cosines.SetElement(i, j, affine[i, j] / spacing[j])
    vtk_image_data.SetDirectionMatrix(direction_cosines)

    return vtk_image_data

def extract_surface(vtk_image_data):
    # Extract surface using the marching cubes algorithm
    surface_extractor = vtk.vtkMarchingCubes()
    surface_extractor.SetInputData(vtk_image_data)
    surface_extractor.SetValue(0, 0.5)  # Threshold for surface extraction, depends on your data
    surface_extractor.Update()
    
    # Optionally, you can smooth the surface
    # smoother = vtk.vtkWindowedSincPolyDataFilter()
    # smoother.SetInputData(surface_extractor.GetOutput())
    # smoother.SetNumberOfIterations(20)
    # smoother.NonManifoldSmoothingOn()
    # smoother.NormalizeCoordinatesOn()
    # smoother.Update()
    
    return surface_extractor.GetOutput()

# Convert numpy array to VTK image data
vtk_image_data = numpy_array_to_vtk_image_data(segmentation_array_list[0])
vtk_image_data = add_affine_information(vtk_image_data, segmentation_affine)

def resample_vtk_image_data(vtk_image_data, reduction_factor):
    """
    Resample the VTK image data to reduce its resolution.

    Parameters:
        vtk_image_data (vtkImageData): The original image data.
        reduction_factor (float): The factor by which to reduce the resolution. For example,
                                  0.5 will reduce the number of points to about half along each dimension.
    Returns:
        vtkImageData: The resampled image data.
    """
    print("A")
    resample = vtk.vtkImageResample()
    print("B")
    resample.SetInputData(vtk_image_data)
    print("C")
    resample.SetAxisMagnificationFactor(0, reduction_factor)
    resample.SetAxisMagnificationFactor(1, reduction_factor)
    resample.SetAxisMagnificationFactor(2, reduction_factor)
    print("D")
    resample.Update()
    print("E")
    return resample.GetOutput()

reduced_vtk_image_data = resample_vtk_image_data(vtk_image_data, 0.1)