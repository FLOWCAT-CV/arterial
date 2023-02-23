#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os, shutil
import vtk

from vtk.util.numpy_support import vtk_to_numpy, numpy_to_vtk

import numpy as np

def perform_centerline_branching(case_dir):
    """
    Performs centerline branching over centerline models. This allows division
    of the centerline tree in segments corresponding to the individual arteries.
    
    This function calls vmtk command vmtkbranchextractor. For additional info refer
    to <http://www.vmtk.org/vmtkscripts/vmtkbranchextractor.html>.

    Saves branched centerline models as:

    >>> case_dir/branch_models/branch_model{idx}.vtk

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 

    Returns
    -------
    
    """
    # List all centerline model to be branched
    centerline_list = sorted([centerline_file for centerline_file in os.listdir(os.path.join(case_dir, "centerlines")) if centerline_file.endswith(".vtk")])
    # Create dir to store all branch_models
    if not os.path.isdir(os.path.join(case_dir, "branch_models")): os.mkdir(os.path.join(case_dir, "branch_models"))

    for idx, centerline_file in enumerate(centerline_list):
        print("Starting branch extraction from centerline model {}...".format(idx))
        # Assing the necessary variables to pass as arguments to vmtkbranchextractor
        centerline_model = os.path.join(case_dir, "centerlines", centerline_file)
        branch_model = os.path.join(case_dir, "branch_models", "branch_model{}.vtk".format(idx))
        radius_array_name = "Radius"
        # Call vmtkbranchextractor command in the command line
        os.system("vmtkbranchextractor -ifile {} -ofile {} -radiusarray {}".format(centerline_model, branch_model, radius_array_name))

    # Unify all clipped models
    perform_branch_model_unification(case_dir)

def perform_branch_model_unification(case_dir):
    """
    Reads all branch_models in the case_dir/branch_models, derived from the case_dir/centerlines files 
    and creates unified branch model. 
    
    Saves unified branch model as:

    >>> case_dir/branch_model.vtk

    Final branch mdoel should have the following cell data arrays updated:
    * centerlinesId -> connections between origin and endpoints
    * tractId -> following a centerline Id, tract number (closest to origin is 0, next is 1 and so on)
    * blanking -> transition to a new branch
    * groupId -> indicates is the centerline is inside of the tract 

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 

    Returns
    -------

    """
    print("Unifying all branch models...")
    branch_model_list = sorted([branch_model_file for branch_model_file in os.listdir(os.path.join(case_dir, "branch_models")) if branch_model_file.endswith(".vtk")])
    # Initialize the vtkPoints and the vtkCellArray objects for the branch_model
    cell_array_branch_model = vtk.vtkCellArray()
    points_branch_model = vtk.vtkPoints()
    # Initialize the 
    final_cell_data_array_branch_model = np.ndarray([4, 0])
    final_radius_array = np.ndarray([1, 0])

    acc_centerline_id = 0
    acc_group_id_branch_model = 0
    points_from_previous_branch_models = 0

    if len(branch_model_list) == 1:
        shutil.copyfile(os.path.join(case_dir, "branch_models", "branch_model0.vtk"), os.path.join(case_dir, "branch_model.vtk"))
    else:
        for branch_model_model_id, _ in enumerate(branch_model_list):
            # Load branch model
            branch_model_path = os.path.join(case_dir, "branch_models", "branch_model{}.vtk".format(branch_model_model_id))
            vtk_poly_data_reader = vtk.vtkPolyDataReader()
            vtk_poly_data_reader.SetFileName(branch_model_path)
            vtk_poly_data_reader.Update()
            branch_model = vtk_poly_data_reader.GetOutput()

            if branch_model.GetNumberOfCells() == 0:
                print("Error in branch model {}. Skipping".format(branch_model_model_id))
            else:
                # Get cell data
                cell_data_array = np.ndarray([4, branch_model.GetNumberOfCells()], dtype=np.int64)
                for idx in range(branch_model.GetCellData().GetNumberOfArrays()):
                    # Get cell data array name
                    cell_data_name = branch_model.GetCellData().GetArrayName(idx)
                    # Add to cell data array
                    cell_data_array[idx] = vtk_to_numpy(branch_model.GetCellData().GetArray(cell_data_name))
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
                radius_array = vtk_to_numpy(branch_model.GetPointData().GetArray("Radius"))

                # On rare occasions, there is a mismatch (a gap) between the number of points of the vtkPolyData and the sum of the number of points from each cell
                # These should be restarted for each branch_modelIdx
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
        final_branch_model = vtk.vtkPolyData()
        final_branch_model.SetPoints(points_branch_model)
        final_branch_model.SetLines(cell_array_branch_model)

        for idx in range(branch_model.GetCellData().GetNumberOfArrays()):
            final_branch_model.GetCellData().AddArray(numpy_to_vtk(final_cell_data_array_branch_model[idx], array_type=vtk.VTK_INT))
            final_branch_model.GetCellData().GetArray(idx).SetName(branch_model.GetCellData().GetArrayName(idx))

        final_branch_model.GetPointData().AddArray(numpy_to_vtk(final_radius_array))
        final_branch_model.GetPointData().GetArray(0).SetName("Radius")

        # Define writer for the vtkPolyData
        writer = vtk.vtkPolyDataWriter()
        writer.SetFileVersion(42)
        writer.SetInputData(final_branch_model)
        writer.SetFileName(os.path.join(case_dir, "branch_model.vtk"))
        writer.Write()

def perform_surface_model_clipping(case_dir):
    """
    Performs clipping over surface models. This allows division
    of the volume model in segments corresponding to the individual arteries.
    
    This function calls vmtk command vmtkbranchclipper. For additional info refer
    to <http://www.vmtk.org/vmtkscripts/vmtkbranchclipper.html>.

    Saves clipped surface models as:

    >>> case_dir/clipped_models/clipped_model{idx}.vtk

    Where idx = 0, 1, ....

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 

    Returns
    -------
    
    """
    # List all surface models to be clipped
    surface_model_list = sorted([surface_model_file for surface_model_file in os.listdir(os.path.join(case_dir, "segmentations")) if surface_model_file.endswith(".vtk")])
    # Create dir to store all clipped_models
    if not os.path.isdir(os.path.join(case_dir, "clipped_models")): os.mkdir(os.path.join(case_dir, "clipped_models"))
    for idx, surface_model_file in enumerate(surface_model_list):
        print("Clipping surface model {}...".format(idx))
        # Assing the necessary variables to pass as arguments to vmtkbranchclipper
        surface_model = os.path.join(case_dir, "segmentations", surface_model_file)
        branch_model = os.path.join(case_dir, "branch_models", "branch_model{}.vtk".format(idx))
        clipped_model = os.path.join(case_dir, "clipped_models", "clipped_model{}.vtk".format(idx))
        radius_array_name = "Radius"
        # Call vmtkbranchclipper command in the command line
        os.system("vmtkbranchclipper -ifile {} -centerlinesfile {} -ofile {} -radiusarray {}".format(surface_model, branch_model, clipped_model, radius_array_name))

    # Unify all clipped models
    perform_clipped_model_unification(case_dir)

def perform_clipped_model_unification(case_dir):
    """
    Reads all  clipped models in case_dir/clipped_models derived from the case_dir/segmentations 
    files and creates unified clipped model. 
    
    Saves unified clipped model as:

    >>> case_dir/clipped_model.vtk

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 

    Returns
    -------

    """
    print("Unifying all clipped models...")
    clipped_model_list = sorted([clipped_model_file for clipped_model_file in os.listdir(os.path.join(case_dir, "clipped_models")) if clipped_model_file.endswith(".vtk")])
    # Initialize the vtkPoints and the vtkCellArray objects for the clipped_model
    cell_array_clipped_model = vtk.vtkCellArray()
    points_clipped_model = vtk.vtkPoints()
    final_group_id_point_array_clipped_model = vtk.vtkIntArray()
    final_group_id_point_array_clipped_model.SetName("GroupIds")

    acc_group_id_clipped_model = 0
    points_from_previous_clipped_models = 0

    if len(clipped_model_list) == 1:
        shutil.copyfile(os.path.join(case_dir, "clipped_models", "clipped_model0.vtk"), os.path.join(case_dir, "clipped_model.vtk"))
    else:
        for clipped_model_id, _ in enumerate(clipped_model_list):
            # Load clipped model         
            clipped_model_path = os.path.join(case_dir, "clipped_models", "clipped_model{}.vtk".format(clipped_model_id))
            vtk_poly_data_reader = vtk.vtkPolyDataReader()
            vtk_poly_data_reader.SetFileName(clipped_model_path)
            vtk_poly_data_reader.Update()
            clipped_model = vtk_poly_data_reader.GetOutput()

            if clipped_model.GetNumberOfCells() == 0:
                print("Error in clipped model {}. Skipping".format(clipped_model_id))
            else:
                print("Processing clipped model {}...".format(clipped_model_id))
                # Get point data (we only get groupId)
                group_id_point_array_clipped_model = vtk_to_numpy(clipped_model.GetPointData().GetArray("GroupIds"))
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

        writer = vtk.vtkPolyDataWriter()
        writer.SetFileVersion(42)
        writer.SetInputData(final_clipped_model)
        writer.SetFileName(os.path.join(case_dir, "clipped_model.vtk"))
        writer.Write()