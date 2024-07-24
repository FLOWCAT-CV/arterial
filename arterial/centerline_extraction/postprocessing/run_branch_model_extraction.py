import sys, pickle

from vmtk import vtkvmtk

from arterial.io.load_and_save_operations import *

def run_branch_extractor_subprocessing(serialized_centerlines_model, blanking_array_name, radius_array_name, group_ids_array_name, centerline_ids_array_name, tract_ids_array_name):
    """
    Auxiliary function to run the centerline branching in a subprocess. This is necessary to avoid
    uncathchable crashes of the underlying VMTK library, which handles processing in C++ and where
    crashes usually result in unrecoverable segmentation faults, causing the main process to fail.

    As a result, the function is called in a subprocess, and the results are stored in a buffer.

    Parameters
    ----------
    queue : multiprocessing.Queue
        Queue to store the results.
    serialized_centerlines_model : string
        Serialized centerlines model.
    blanking_array_name : string
        Name of the blanking array.
    radius_array_name : string
        Name of the radius array.
    group_ids_array_name : string
        Name of the group ids array.
    centerline_ids_array_name : string
        Name of the centerline ids array.
    tract_ids_array_name : string
        Name of the tract ids array

    Returns
    -------

    """
    try:
        centerlines_model = deserialize_vtk_polydata(serialized_centerlines_model)
        branchExtractor = vtkvmtk.vtkvmtkCenterlineBranchExtractor()
        branchExtractor.SetInputData(centerlines_model)
        branchExtractor.SetBlankingArrayName(blanking_array_name)
        branchExtractor.SetRadiusArrayName(radius_array_name)
        branchExtractor.SetGroupIdsArrayName(group_ids_array_name)
        branchExtractor.SetCenterlineIdsArrayName(centerline_ids_array_name)
        branchExtractor.SetTractIdsArrayName(tract_ids_array_name)
        branchExtractor.Update()
        branch_model = branchExtractor.GetOutput()
        result = serialize_vtk_polydata(branch_model)
        print("    Centerline branching completed.")
    except Exception as e:
        print(f"    Centerline branching failed: {e}. \nThis is most likely a VMTK issue. \nIf this is the first model (idx=0) " \
              "the process will be interrupted, otherwise, the process will continue, ignoring the failed model "\
              "(Usually the first one is the largest and most relevant).")
        result = None
    print("    Subprocess completed execution.")

    # Output result as bytes
    sys.stdout.buffer.write(pickle.dumps(result))
    sys.exit(0 if result else 1)

if __name__ == "__main__":
    # Reading input arguments
    input_data = sys.stdin.buffer.read()
    serialized_centerlines_model, blanking_array_name, radius_array_name, group_ids_array_name, centerline_ids_array_name, tract_ids_array_name = pickle.loads(input_data)

    # Call function with unpacked arguments
    run_branch_extractor_subprocessing(
        serialized_centerlines_model,
        blanking_array_name,
        radius_array_name,
        group_ids_array_name,
        centerline_ids_array_name,
        tract_ids_array_name
    )