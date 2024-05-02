import os
from arterial.io.load_and_save_operations import *
from prova_centerline_extraction import compute_network_centerlines, get_endpoints, aortic_arch_endpoint_check, multi_robust_endpoint_detection, robust_endpoint_detection

from time import time

case_dir = "/home/vhir/github/arterial/tests/test_data/output"
mode = "extracranial_vessels"

segmentation = load_vtkpolydata(os.path.join(case_dir, mode, "segmentation.vtk"))
segmentation_nifti = load_nifti(os.path.join(case_dir, mode, "segmentation.nii.gz"))
segmentation_array = segmentation_nifti.get_fdata()
segmentation_affine = segmentation_nifti.affine

start = time()
print("Extracting centerline network...")
network_centerlines = compute_network_centerlines(segmentation)
timestamp_0 = time()
print("\nCenterline network extracted in {:.2f} seconds".format(timestamp_0 - start))
print("Extracting endpoints...")
endpoints = get_endpoints(network_centerlines, None)
for idx, endpoint in enumerate(endpoints):
    endpoints[idx] = np.array(endpoint)
timestamp_1 = time()
print("Endpoints extracted in {:.2f} seconds".format(timestamp_1 - timestamp_0))

print("Checking presence of AA endpoints...")
endpoints = aortic_arch_endpoint_check(endpoints, segmentation_array, segmentation_affine)
timestamp_2 = time()
print("Aortic arch endpoints checked in {:.2f} seconds".format(timestamp_2 - timestamp_1))

print("Robust endpoint replacement...")
# endpoints = robust_endpoint_detection(endpoints, segmentation_array, segmentation_affine, window_size=10)
endpoints = multi_robust_endpoint_detection(endpoints, segmentation_array, segmentation_affine, window_size=10, max_workers=10)
timestamp_3 = time()
print("Robust endpoint detection in {:.2f} seconds".format(timestamp_3 - timestamp_2))

print("Total time: {:.2f} seconds".format(timestamp_3 - start))

def get_robuts_endpoints(segmentation):
    """
    Computed a set of robust endpoints for a segmentation surface model. It first
    computes the centerline network of the segmentation and then extracts the endpoints.
    The endpoints are then checked for the presence of the aortic arch and replaced with
    more robust endpoints.

    These functions intend to replicate the behiavior of the auto-detect endpoints function
    from the VKTK Slicer extension. 

    Parameters
    ----------
    segmentation : vtkPolyData
        The segmentation surface model.

    Returns
    -------
    endpoints : list
        A list of 3D ijk coordinates of the robust endpoints.

    """
    # Computes the centerline network of the segmentation
    network_centerlines = compute_network_centerlines(segmentation)
    # Extracts the endpoints (direct implementation of the getEndPoints function of the VKTK Slicer extension)
    endpoints = getEndPoints(network_centerlines, None)
    # Converts the endpoints to numpy arrays
    for idx, endpoint in enumerate(endpoints):
        endpoints[idx] = np.array(endpoint)
    # Checks the presence of the aortic arch endpoints
    endpoints = aortic_arch_endpoint_check(endpoints, segmentation_array, segmentation_affine)
    # Computes the robust endpoints. This helps avoid centerline extraction errors due to the endpoints being outside the segmentation
    endpoints = multi_robust_endpoint_detection(endpoints, segmentation_array, segmentation_affine, window_size=10, max_workers=10)

    return endpoints