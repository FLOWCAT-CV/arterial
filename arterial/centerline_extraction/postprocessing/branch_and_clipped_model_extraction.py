#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import vtk

from vmtk import vtkvmtk

import numpy as np

import pickle
import subprocess
from arterial.io.load_and_save_operations import *

def extract_branch_model(centerlines_model, blanking_array_name="Blanking", radius_array_name="MaximumInscribedSphereRadius", group_ids_array_name="GroupIds", centerline_ids_array_name="CenterlineIds", tract_ids_array_name="TractIds"):
    """
    Performs centerline branching over centerline models. This allows division
    of the centerline tree in segments corresponding to the individual arteries.
    
    This function summons the vtkvmtk.vtkvmtkCenterlineBranchExtractor() class. 
    For additional info refer to <https://github.com/vmtk/vmtk/blob/master/vmtkScripts/vmtkbranchextractor.py>.

    Parameters
    ----------
    centerlines_model : vtkPolyData
        Centerlines model. 
    blanking_array_name : string, optional
        Name of the blanking array. The default is "Blanking".
    radius_array_name : string, optional
        Name of the radius array. The default is "MaximumInscribedSphereRadius".
    group_ids_array_name : string, optional
        Name of the group ids array. The default is "GroupIds".
    centerline_ids_array_name : string, optional
        Name of the centerline ids array. The default is "CenterlineIds".
    tract_ids_array_name : string, optional
        Name of the tract ids array. The default is "TractIds".

    Returns
    -------
    branch_model : vtkPolyData
        Branched centerline model.
    
    """
    try:
        # Serialize centerlines_model (dummy example - replace with actual serialization if needed)
        serialized_centerlines_model = serialize_vtk_polydata(centerlines_model)

        # Start the subprocess
        proc = subprocess.Popen(
            ['python3', os.path.join(os.path.abspath(""), 'arterial/centerline_extraction/postprocessing/run_branch_model_extraction.py')],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )

        args = (blanking_array_name, radius_array_name, group_ids_array_name, centerline_ids_array_name, tract_ids_array_name)  

        # Send serialized data to subprocess
        input_data = pickle.dumps((serialized_centerlines_model, *args))
        stdout_data, stderr_data = proc.communicate(input=input_data)

        if proc.returncode != 0:
            error_message = stderr_data.decode()
            print(f"Subprocess failed with error: {error_message}")
            return None

        # Deserialize the result
        result_str = pickle.loads(stdout_data)
        if result_str:
            # Deserialize vtkPolyData
            branch_model = deserialize_vtk_polydata(result_str)
            return branch_model
        else:
            return None

    except Exception as e:
        print(f"Exception occurred in branch model extraction: {e}")
        return None
    

# def extract_branch_model(centerlines_model, blanking_array_name="Blanking", radius_array_name="MaximumInscribedSphereRadius", group_ids_array_name="GroupIds", centerline_ids_array_name="CenterlineIds", tract_ids_array_name="TractIds"):
#     """
#     Performs centerline branching over centerline models. This allows division
#     of the centerline tree in segments corresponding to the individual arteries.
    
#     This function summons the vtkvmtk.vtkvmtkCenterlineBranchExtractor() class. 
#     For additional info refer to <https://github.com/vmtk/vmtk/blob/master/vmtkScripts/vmtkbranchextractor.py>.

#     Parameters
#     ----------
#     centerlines_model : string or path-like object
#         Path to centerlines model. 
#     blanking_array_name : string, optional
#         Name of the blanking array. The default is "Blanking".
#     radius_array_name : string, optional
#         Name of the radius array. The default is "MaximumInscribedSphereRadius".
#     group_ids_array_name : string, optional
#         Name of the group ids array. The default is "GroupIds".
#     centerline_ids_array_name : string, optional
#         Name of the centerline ids array. The default is "CenterlineIds".
#     tract_ids_array_name : string, optional
#         Name of the tract ids array. The default is "TractIds".

#     Returns
#     -------
#     branch_model : vtkPolyData
#         Branched centerline model.
    
#     """
#     # Initialize the vtkvmtkCenterlineBranchExtractor object
#     branchExtractor = vtkvmtk.vtkvmtkCenterlineBranchExtractor()
#     branchExtractor.SetInputData(centerlines_model)
#     branchExtractor.SetBlankingArrayName(blanking_array_name)
#     branchExtractor.SetRadiusArrayName(radius_array_name)
#     branchExtractor.SetGroupIdsArrayName(group_ids_array_name)
#     branchExtractor.SetCenterlineIdsArrayName(centerline_ids_array_name)
#     branchExtractor.SetTractIdsArrayName(tract_ids_array_name)

#     # Execute the branch extraction
#     try:
#         branchExtractor.Update()
#         branch_model = branchExtractor.GetOutput()
#     except:
#         print("Centerline branching failed. This is a VMTK issue. \nIf this is the first model (idx=0) " \
#               "the process will be interrupted, otherwise, the process will continue, ignoring the failed model "\
#               "(Usually the first one is the largest and most relevant).")
#         branch_model = None

#     return branch_model

def unify_branch_models(branch_model_list, radius_array_name="MaximumInscribedSphereRadius"):
    """
    Unifies all branch models in a single vtkPolyData object.

    Final branch model should have the following cell data arrays updated:
    * centerlinesId -> connections between origin and endpoints
    * tractId -> following a centerline Id, tract number (closest to origin is 0, next is 1 and so on)
    * blanking -> transition to a new branch
    * groupId -> indicates is the centerline is inside of the tract 

    Parameters
    ----------
    branch_model_list : list
        List of branch models.  
    radius_array_name : string, optional
        Name of the radius array. The default is "MaximumInscribedSphereRadius".

    Returns
    -------
    unified_branch_model : vtkPolyData
        Unified branch model.

    """
    if len(branch_model_list) == 1:
        return branch_model_list[0]
    else:
        # Initialize the vtkPoints and the vtkCellArray objects for the branch_model
        cell_array_branch_model = vtk.vtkCellArray()
        points_branch_model = vtk.vtkPoints()
        # Initialize the cell data arrays for the branch_model
        final_cell_data_array_branch_model = np.ndarray([4, 0])
        final_radius_array = np.ndarray([1, 0])

        acc_centerline_id = 0
        acc_group_id_branch_model = 0
        points_from_previous_branch_models = 0
        for branch_model_idx, branch_model in enumerate(branch_model_list):
            print("Processing branch model {}...".format(branch_model_idx))
            if branch_model.GetNumberOfCells() == 0:
                print("Error in branch model {}. Skipping".format(branch_model_idx))
            else:
                # Get cell data
                cell_data_array = np.ndarray([4, branch_model.GetNumberOfCells()], dtype=np.int64)
                for idx in range(branch_model.GetCellData().GetNumberOfArrays()):
                    # Get cell data array name
                    cell_data_name = branch_model.GetCellData().GetArrayName(idx)
                    # Add to cell data array
                    cell_data_array[idx] = vtk.util.numpy_support.vtk_to_numpy(branch_model.GetCellData().GetArray(cell_data_name))
                    if cell_data_name == "CenterlineIds":
                        # Add accumulated centerline id
                        cell_data_array[idx] += acc_centerline_id
                        # Update accumulated centerlineId and groupId
                        acc_centerline_id += np.amax(cell_data_array[idx]) + 1
                    elif cell_data_name == "GroupIds":
                        # Add accumulated group id
                        cell_data_array[idx] += acc_group_id_branch_model
                        # Update accumulated centerlineId and groupId
                        acc_group_id_branch_model += np.amax(cell_data_array[idx]) + 1

                # Append cell_data_array from present branch_model
                final_cell_data_array_branch_model = np.append(final_cell_data_array_branch_model, cell_data_array, axis=1)
                
                # Get point data (we only get radius)
                radius_array = vtk.util.numpy_support.vtk_to_numpy(branch_model.GetPointData().GetArray(radius_array_name))

                # On rare occasions, there is a mismatch (a gap) between the number of points of the vtkPolyData and the sum of the number of points from each cell
                # These should be restarted for each branch_model_idx
                gap = 0
                idx_points_minus_gap = 0

                for idx in range(branch_model.GetNumberOfCells()):
                    poly_line = branch_model.GetCell(idx)
                    new_poly_line = vtk.vtkPolyLine()
                    new_poly_line_points = vtk.vtkPoints()
                    new_poly_line_points_ids = []
                    for idx2 in range(poly_line.GetNumberOfPoints()):
                        # Condition tells us if there is a diference between current cell point and branch_model point with accumulated gap
                        condition = np.abs(np.sum(np.array(poly_line.GetPoints().GetPoint(idx2)) - np.array(branch_model.GetPoints().GetPoint(idx_points_minus_gap + gap)))) < 0.01
                        while not condition:
                            # Insert point in vtkPoints
                            points_branch_model.InsertNextPoint(branch_model.GetPoints().GetPoint(idx_points_minus_gap + gap))
                            # Insert point in new_poly_line
                            new_poly_line_points.InsertNextPoint(branch_model.GetPoints().GetPoint(idx_points_minus_gap + gap))
                            # Get radius data for each point
                            final_radius_array = np.append(final_radius_array, radius_array[idx_points_minus_gap + gap])
                            # Change point id for each point in the cell, taking into account accumulated number of points
                            new_poly_line_points_ids.append(idx_points_minus_gap + points_from_previous_branch_models + gap)
                            # Update gap
                            gap += 1
                            # Recompute condition
                            condition = np.abs(np.sum(np.array(poly_line.GetPoints().GetPoint(idx2)) - np.array(branch_model.GetPoints().GetPoint(idx_points_minus_gap + gap)))) < 0.01

                        # Insert point in vtkPoints
                        points_branch_model.InsertNextPoint(branch_model.GetPoints().GetPoint(idx_points_minus_gap + gap))
                        # Insert point in new_poly_line
                        new_poly_line_points.InsertNextPoint(branch_model.GetPoints().GetPoint(idx_points_minus_gap + gap))
                        # Get radius data for each point
                        final_radius_array = np.append(final_radius_array, radius_array[idx_points_minus_gap + gap])
                        # Change point id for each point in the cell, taking into account accumulated number of points
                        new_poly_line_points_ids.append(idx_points_minus_gap + points_from_previous_branch_models + gap)
                        # Update idx_points_minus_gap
                        idx_points_minus_gap += 1   
                    
                    new_poly_line.Initialize(len(new_poly_line_points_ids), new_poly_line_points_ids, new_poly_line_points)  
                    # Insert cell in vtkCellArray
                    cell_array_branch_model.InsertNextCell(new_poly_line)
                
                # Update total number of points from previous models
                points_from_previous_branch_models += branch_model.GetNumberOfPoints()
                
        # Store all branch model data in new vtkPolyData
        unified_branch_model = vtk.vtkPolyData()
        unified_branch_model.SetPoints(points_branch_model)
        unified_branch_model.SetLines(cell_array_branch_model)

        for idx in range(branch_model.GetCellData().GetNumberOfArrays()):
            unified_branch_model.GetCellData().AddArray(vtk.util.numpy_support.numpy_to_vtk(final_cell_data_array_branch_model[idx], array_type=vtk.VTK_INT))
            unified_branch_model.GetCellData().GetArray(idx).SetName(branch_model.GetCellData().GetArrayName(idx))

        unified_branch_model.GetPointData().AddArray(vtk.util.numpy_support.numpy_to_vtk(final_radius_array))
        unified_branch_model.GetPointData().GetArray(0).SetName(radius_array_name)

        return unified_branch_model
    
def extract_clipped_model(surface_model, branch_model, blanking_array_name="Blanking", radius_array_name="MaximumInscribedSphereRadius", group_ids_array_name="GroupIds"):
    """
    Performs clipping over surface models. This allows division
    of the volume model in segments corresponding to the individual arteries.
    
    This function summons the vtkvmtk.vtkvmtkPolyDataCenterlineGroupsClipper() class. 
    For additional info refer to <https://github.com/vmtk/vmtk/blob/master/vmtkScripts/vmtkbranchclipper.py>.

    For now, the branch clipper model does not work reliably, so we will not be using this 
    function as part of the vanilla pipeline.

    Parameters
    ----------
    surface_model : string or path-like object
        Path to surface model. 
    branch_model : string or path-like object
        Path to branch model. 
    blanking_array_name : string, optional
        Name of the blanking array. The default is "Blanking".
    radius_array_name : string, optional
        Name of the radius array. The default is "MaximumInscribedSphereRadius".
    group_ids_array_name : string, optional
        Name of the group ids array. The default is "GroupIds".

    Returns
    -------
    clipped_model : vtkPolyData
        Clipped surface model.
    
    """
    # Initialize the vtkvmtkBranchClipper object
    branchClipper = vtkvmtk.vtkvmtkPolyDataCenterlineGroupsClipper()
    branchClipper.SetInputData(surface_model)
    branchClipper.SetCenterlines(branch_model)
    branchClipper.SetBlankingArrayName(blanking_array_name)
    branchClipper.SetCenterlineRadiusArrayName(radius_array_name)
    branchClipper.SetCenterlineGroupIdsArrayName(group_ids_array_name)
    branchClipper.SetGroupIdsArrayName(group_ids_array_name)
    branchClipper.SetCutoffRadiusFactor(0.)
    branchClipper.SetClipValue(1.)
    branchClipper.SetUseRadiusInformation(True)
    branchClipper.ClipAllCenterlineGroupIdsOn() # Interesting that you can set a list of group Ids and only apply clipping to those groups. See https://github.com/vmtk/vmtk/blob/master/vmtkScripts/vmtkbranchclipper.py for the recipe

    # Execute the branch clipping
    try:
        branchClipper.Update()
        clipped_model = branchClipper.GetOutput()
    except:
        print("Clipping failed. This is a VMTK issue. \nIf this is the first model (idx=0) " \
              "the process will be interrupted, otherwise, the process will continue, ignoring the failed model "\
              "(Usually the first one is the largest and most relevant).")
        clipped_model = None

    return clipped_model

def unify_clipped_models(clipped_model_list):
    """
    Unifies all clipped models in a single vtkPolyData object.

    Parameters
    ----------
    clipped_model_list : list
        List of clipped models.

    Returns
    -------
    final_clipped_model : vtkPolyData
        Unified clipped model.

    """
    if len(clipped_model_list) == 1:
        return clipped_model_list[0]
    else:
        # Initialize the vtkPoints and the vtkCellArray objects for the clipped_model
        cell_array_clipped_model = vtk.vtkCellArray()
        points_clipped_model = vtk.vtkPoints()
        final_group_id_point_array_clipped_model = vtk.vtkIntArray()
        final_group_id_point_array_clipped_model.SetName("GroupIds")

        acc_group_id_clipped_model = 0
        points_from_previous_clipped_models = 0
        for clipped_model_idx, clipped_model in enumerate(clipped_model_list):
            if clipped_model.GetNumberOfCells() == 0:
                print("Error in clipped model {}. Skipping".format(clipped_model_idx))
            else:
                print("Processing clipped model {}...".format(clipped_model_idx))
                # Get point data (we only get groupId)
                group_id_point_array_clipped_model = vtk.numpy_support.vtk_to_numpy(clipped_model.GetPointData().GetArray("GroupIds"))
                # Update groupIds of current clipped_model
                group_id_point_array_clipped_model = group_id_point_array_clipped_model + acc_group_id_clipped_model
                # Update accumulated groupId
                acc_group_id_clipped_model += np.amax(group_id_point_array_clipped_model) + 1
                # Get number of points in each clipped model cell (= 3)
                number_of_point_ids = clipped_model.GetCell(0).GetPointIds().GetNumberOfIds()
                # Generally, points are placed as cell indices go up, but this is not always the case
                # To speed up computations, we only search for cells with higher cellIds than the ones already searched for, 
                # But in the cases where a point_idx has not been found, we search across all cells of the model, in order
                # to ensure that no point_idx is missed
                last_cell = 0
                # We iterate through every pointId
                for point_idx in range(clipped_model.GetNumberOfPoints()):
                    # We need a boolean variable to stop the iterative search when a point is found to speed up computations
                    found_point = False
                    # We primarily only search for cells with a cellId larger than the ones analyzed
                    # Limiting up the search dramatically speeds up computations
                    for cell_idx in range(max(0, last_cell - 1), clipped_model.GetNumberOfCells()):
                        # Iterate over points in cell
                        for idx in range(number_of_point_ids):
                            # If a point is found with pointId equal to the next point_idx
                            if clipped_model.GetCell(cell_idx).GetPointId(idx) == point_idx:
                                # Keep cell_idx to limit cell of the next point_idx
                                last_cell = cell_idx
                                # Insert next point in final clipped model point object and groupId point array
                                points_clipped_model.InsertNextPoint(clipped_model.GetCell(cell_idx).GetPoints().GetPoint(idx))
                                final_group_id_point_array_clipped_model.InsertNextValue(group_id_point_array_clipped_model[clipped_model.GetCell(cell_idx).GetPointId(idx)])
                                # Update boolean marker to stop the search for the current pointidx
                                found_point = True
                                break
                        # Break cell serach if point is found
                        if found_point:
                            break
                    # If point is not found, search all throughout the cell pool, including cells with a smaller cell_idx than last_cell
                    # These searches are significantly longer than the general case, but we only apply them when needed
                    # This is very rare but if not done, it will mess up the final model
                    if not found_point:
                        # If point_idx has not been found, we also look at the previous cells (rare but it happens)
                        for cell_idx in range(clipped_model.GetNumberOfCells()):
                            # Iterate over points in cell
                            for idx in range(number_of_point_ids):
                                # If a point is found with pointId equal to the next point_idx
                                if clipped_model.GetCell(cell_idx).GetPointId(idx) == point_idx:
                                    # Keep cell_idx to limit cell of the next point_idx
                                    last_cell = cell_idx
                                    # Insert next point in final clipped model point object and groupId point array
                                    points_clipped_model.InsertNextPoint(clipped_model.GetCell(cell_idx).GetPoints().GetPoint(idx))
                                    final_group_id_point_array_clipped_model.InsertNextValue(group_id_point_array_clipped_model[clipped_model.GetCell(cell_idx).GetPointId(idx)])
                                    # Update boolean marker to stop the search for the current pointidx
                                    found_point = True
                            # Break cell serach if point is found
                            if found_point:
                                break

                # We need this to set the new pointIds for the triangles with the SetId method. This will be 3
                # Insert the cells with the corresponding groupId to the new vtkCellArray
                # for idx in cellIdArray:
                for cell_idx in range(clipped_model.GetNumberOfCells()):
                    cell = vtk.vtkTriangle()
                    cell.GetPointIds().SetNumberOfIds(number_of_point_ids)
                    for idx in range(number_of_point_ids):
                        cell.GetPointIds().SetId(idx, clipped_model.GetCell(cell_idx).GetPointId(idx) + points_from_previous_clipped_models)
                    cell_array_clipped_model.InsertNextCell(cell)
                
                # Update total number of points from previous models
                points_from_previous_clipped_models += clipped_model.GetNumberOfPoints()

        # Store all clipped model data in new vtkPolyData
        final_clipped_model = vtk.vtkPolyData()
        final_clipped_model.SetPoints(points_clipped_model)
        final_clipped_model.SetPolys(cell_array_clipped_model)
        final_clipped_model.GetPointData().AddArray(final_group_id_point_array_clipped_model)

        # We can to compute the normals for all mesh triangles
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(final_clipped_model)
        normals.SetFeatureAngle(80)
        normals.AutoOrientNormalsOn()
        normals.UpdateInformation()
        normals.Update()
        final_clipped_model = normals.GetOutput()

        # We also pass a clean vtkPolyData filter for good measure
        clean_poly_data = vtk.vtkCleanPolyData()
        clean_poly_data.SetInputData(final_clipped_model)
        clean_poly_data.Update()
        final_clipped_model = clean_poly_data.GetOutput()

        return final_clipped_model