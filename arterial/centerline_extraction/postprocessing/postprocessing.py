#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os, shutil
import vtk

import numpy as np
import nibabel as nib

def compute_centerline_segments_array(case_dir):
    ''' 
    Loads a vtkPolyData object containing the centerline model and generates
    centerline_segments_array, spliting the centerline cells into individual 
    segments between bifurcations, associating a new identifyier to them. From 
    each segment, it stores centerine point coordinates and maximal inscribed 
    sphere radius. Coordinates in the resulting array are in mm, subtracting the
    tranlation for each case.

    The output is piped to the generation of the corresponding graph. Saves centerline_segments_array as a 
    .npy file:

    >>> case_dir/centerline_segments_array.npy

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory.

    Returns
    -------
        
    '''
    centerline_list = sorted([centerline_file for centerline_file in os.listdir(os.path.join(case_dir, "centerlines")) if centerline_file.endswith(".vtk")])
    final_centerline_segments_array = np.ndarray([0, 2])

    for centerline_idx, _ in enumerate(centerline_list):
        # Load centerlines.vtk as a vtkPolyData object
        centerline_poly_data_reader = vtk.vtkPolyDataReader()
        centerline_poly_data_reader.SetFileName(os.path.join(case_dir, "centerlines", "centerlines{}.vtk".format(centerline_idx)))
        centerline_poly_data_reader.Update()
        centerline_model = centerline_poly_data_reader.GetOutput()
    
        # Define number of cells and points in the vtkPolyData
        number_of_cells = centerline_model.GetNumberOfCells()

        # Declare empty arrays. We will store radius and coordinated for each centerline in arrays corresponding to each cell (each centerline path) 
        # that will be variable in size. In turn, we will store these in super arrays for each centerline model and join them at the end
        cells_id_array = np.ndarray([number_of_cells], dtype=int)
        cells_coordinate_array = np.ndarray([number_of_cells], dtype=object)
        cells_radius_array = np.ndarray([number_of_cells], dtype=object)
        length_coordinate_array = np.ndarray([number_of_cells], dtype=int)

        # Define affine matrix and invert
        aff = nib.load(os.path.join(case_dir, "{}_segmentation.nii.gz".format(os.path.basename(case_dir)))).affine
        # Compute translation from affine matrix
        translation = np.transpose(aff[:3, 3])

        # Centerline point data for maximal inscribed sphere radius
        radius_array = vtk.util.numpy_support.vtk_to_numpy(centerline_model.GetPointData().GetArray(0))

        # Iterate over cells to extract cell_ids, positions and radii. We also store lengths of cells (number of points, not distance)
        for cell_id in range(number_of_cells):
            cells_id_array[cell_id] = cell_id
            cell = vtk.vtkGenericCell()
            centerline_model.GetCell(cell_id, cell)
            number_of_cell_points = cell.GetNumberOfPoints()
            cells_coordinate_array[cell_id] = np.ndarray([number_of_cell_points, 3])
            # In some cases the pointIds of the centerline points are reversed
            if cell.GetPointId(cell.GetPointIds().GetNumberOfIds() - 1) < cell.GetPointId(0):
                cells_radius_array[cell_id] = np.flip(radius_array[cell.GetPointId(cell.GetPointIds().GetNumberOfIds() - 1):cell.GetPointId(0) + 1])
            else:
                cells_radius_array[cell_id] = np.flip(radius_array[cell.GetPointId(0) - 1:cell.GetPointId(cell.GetPointIds().GetNumberOfIds() - 1)])
            # We subtract the translation from the affine matrix to the coordinates and invert the sign of the first to make all coordinates positive
            # The resulting coordinates system is a translation-less 
            for idx in range(number_of_cell_points):
                # Subtract translation
                cells_coordinate_array[cell_id][idx] = cell.GetPoints().GetPoint(idx) - translation
                # Change the sign of the first component of the coordinate (L ro R)
                cells_coordinate_array[cell_id][idx][0] = - cells_coordinate_array[cell_id][idx][0]
            length_coordinate_array[cell_id] = number_of_cell_points

        # Now, we start analyzing each centerline starting from the shortest to the longest. Here we want to analyze overlap between
        # the different centerline cells to generate arrays that only contain centerline points from independent segments, without overlap
        from_shortest_to_longest = np.sort(length_coordinate_array)
        aux_from_shortest_to_longest = np.argsort(length_coordinate_array) # Auxiliar array that will be iteratively deleted
        bifurcations_array = np.ndarray([0, 2], dtype=int)

        # We iterate over all segments starting from the shortest segment, except for the longest (unnecesary)
        for shortest_length in from_shortest_to_longest[:-1]:
            # New array created for this iteration. Only contains segments with longer length than currently analyzed, 
            # and up to the length of the segment. The goal is to identify bifurcations along each curve.
            aux_array = np.ndarray([len(aux_from_shortest_to_longest), shortest_length, 3]) 
            # We order them from shortest to longest only in the aux_array
            idx_aux = 0 
            for idx1 in aux_from_shortest_to_longest:
                aux_array[idx_aux] = cells_coordinate_array[idx1][:shortest_length]
                idx_aux += 1

            for idx1 in range(len(aux_from_shortest_to_longest)):
                cell_id = aux_from_shortest_to_longest[idx1]
                # cell_id indicates cell_id of an alternative cell
                aux_array_2 = aux_array # We generate a second copy of the auxiliar array that we delete iteratively
                # We iterate over coordinates of each cell with a lenght longer or equal than
                # the current shortest cell, searching for bifurcation points
                for coord_idx in range(shortest_length):
                    # coord_idx indicates point position on the current cell
                    # For each, coordinate, we iterate over all other cells to try to locate a bifurcation
                    for idx2 in range(1, aux_array_2.shape[0]): # We iterate over the cells that are still on the path
                        # idx2 indicates cell_id of an alternative cell
                        if not (aux_array_2[0][coord_idx] == aux_array_2[idx2][coord_idx]).all():
                        # if np.linalg.norm(aux_array_2[0][coord_idx] - aux_array_2[idx2][coord_idx]) > 10:
                            bifurcations_array = np.append(bifurcations_array, [[cell_id, coord_idx - 1]], 0)
                            # Only continue with branches overlapping with current cell_id
                            aux_array_2 = np.delete(aux_array_2, np.where(aux_array_2[:, coord_idx, 0] != aux_array_2[0, coord_idx, 0]), axis=0)
                            # Skip to next point
                            break

                aux_array = np.roll(aux_array, -1, axis=0)

            # When all positions have been covered, get rid of the shortest cell and repeat with the next one
            aux_from_shortest_to_longest = aux_from_shortest_to_longest[1:]

        if bifurcations_array.size == 0:
            bifurcations_array = np.array([[0, 0]], dtype=int)

        bifurcations_array = np.unique(bifurcations_array, axis=0)

        segments_position_array = np.ndarray([0, 3], dtype=int)

        # Now we generate the segments from the bifurcation points
        for cell_id in range(number_of_cells):
            aux_bifurcations_id = np.squeeze(np.array(np.where(bifurcations_array[:, 0] == cell_id)), axis=0)
            for idx in range(len(aux_bifurcations_id)):
                if idx == 0: # First segment of the cell
                    segments_position_array = np.append(segments_position_array, [[cell_id, 0, bifurcations_array[aux_bifurcations_id[idx], 1]]], axis=0)
                else: # Segments in the middle
                    segments_position_array = np.append(segments_position_array, [[cell_id, bifurcations_array[aux_bifurcations_id[idx-1], 1], bifurcations_array[aux_bifurcations_id[idx], 1]]], axis=0)
                if idx == len(aux_bifurcations_id) - 1: # Last segment of the cell
                    segments_position_array = np.append(segments_position_array, [[cell_id, bifurcations_array[aux_bifurcations_id[idx], 1], -1]], axis=0)

        # We can find the start- and endpoints in ijk (voxel) coordinates
        segments_coordinate_array = np.ndarray([len(segments_position_array), 2, 3])

        for idx in range(len(segments_coordinate_array)):
            segments_coordinate_array[idx, 0] = cells_coordinate_array[segments_position_array[idx, 0]][segments_position_array[idx, 1]]
            segments_coordinate_array[idx, 1] = cells_coordinate_array[segments_position_array[idx, 0]][segments_position_array[idx, 2]]

        # With the IJK coordinates, we can eliminate duplicate segments
        _, order = np.unique(segments_coordinate_array, return_index=True, axis=0)
        unique_segments_position_array = segments_position_array[np.sort(order)]

        # We create the centerline_segments_array that will contain the non-overlapping cells
        centerline_segments_array = np.ndarray([len(unique_segments_position_array), 2], dtype=object)

        for idx in range(len(unique_segments_position_array)):
            if unique_segments_position_array[idx, 2] == -1:
                # We have to make a distinction for the cases that end at an endpoint, rather than at a bifurcation
                centerline_segments_array[idx, 0] = cells_coordinate_array[unique_segments_position_array[idx, 0]][unique_segments_position_array[idx, 1]:]
                centerline_segments_array[idx, 1] = cells_radius_array[unique_segments_position_array[idx, 0]][unique_segments_position_array[idx, 1]:]
            else:    
                centerline_segments_array[idx, 0] = cells_coordinate_array[unique_segments_position_array[idx, 0]][unique_segments_position_array[idx, 1]:unique_segments_position_array[idx, 2] + 1]
                centerline_segments_array[idx, 1] = cells_radius_array[unique_segments_position_array[idx, 0]][unique_segments_position_array[idx, 1]:unique_segments_position_array[idx, 2] + 1]

        # Add filter for repeated points
        for idx, segment in enumerate(centerline_segments_array):
            remove_point_idx = []
            # We look at consecutive points. If two consecutive points are the same, we remove one of them
            for point_idx in range(1, len(segment[0])):
                if np.linalg.norm(segment[0][point_idx - 1] - segment[0][point_idx]) < 1e-4:
                    remove_point_idx.append(point_idx)
            centerline_segments_array[idx, 0] = np.delete(centerline_segments_array[idx, 0], remove_point_idx)
            centerline_segments_array[idx, 1] = np.delete(centerline_segments_array[idx, 1], remove_point_idx)

        remove_floating = []
        for idx, segment in enumerate(centerline_segments_array):
            if len(segment[0]) < 2:
                remove_floating.append(idx)

        centerline_segments_array = np.delete(centerline_segments_array, remove_floating, axis = 0)

        # Check if circular segments should be joint
        new_segments = []
        new_segments_radius = []
        delete_idx = []
        for idx1 in range(len(centerline_segments_array)):
            for idx2 in range(len(centerline_segments_array)):
                if (centerline_segments_array[idx1, 0][-1] == centerline_segments_array[idx2, 0][-1]).all() and idx1 != idx2 and idx1 not in delete_idx:
                    # Shortest segment should be joint at the end of the other one
                    if len(centerline_segments_array[idx1, 0]) < len(centerline_segments_array[idx2, 0]): # idx shorter
                        new_segments.append(np.append(centerline_segments_array[idx2, 0], centerline_segments_array[idx1, 0], axis = 0))
                        new_segments_radius.append(np.append(centerline_segments_array[idx2, 1], centerline_segments_array[idx1, 1]))
                    else:
                        new_segments.append(np.append(centerline_segments_array[idx1, 0], centerline_segments_array[idx2, 0], axis = 0))
                        new_segments_radius.append(np.append(centerline_segments_array[idx1, 1], centerline_segments_array[idx2, 1]))
                    delete_idx.append(idx1)
                    delete_idx.append(idx2)

        centerline_segments_array = np.delete(centerline_segments_array, delete_idx, axis = 0)

        for idx in range(len(new_segments)):
            new_segment = np.ndarray([1, 2], dtype = object)
            new_segment[0, 0] = new_segments[idx]
            new_segment[0, 1] = new_segments_radius[idx]
            centerline_segments_array = np.append(centerline_segments_array, new_segment, axis=0)
            
        final_centerline_segments_array = np.append(final_centerline_segments_array, centerline_segments_array, axis=0)

    # We compare the startpoint of each cell with all other cell startpoints
    delete_idx = []
    for idx1 in range(len(final_centerline_segments_array)):
        startpoint1 = final_centerline_segments_array[idx1][0][0]
        for idx2 in range(idx1 + 1, len(final_centerline_segments_array)): # We ignore previous cells to avoid repeating comparisons
            startpoint2 = final_centerline_segments_array[idx2][0][0]
            # If startpoints coincide, we can either be in the case of interest (there exists a node with degree = 2)
            # or we can be in a bifurcation. We have to rule out the bifurcation in order to identify this event.
            # To do that, we ensure that this point is not in any other cell
            if np.linalg.norm(startpoint1 - startpoint2) < 1e-4:
                is_bifurcation = False
                for idx3 in range(len(final_centerline_segments_array)):
                    startpoint3 = final_centerline_segments_array[idx3][0][0]
                    endpoint3 = final_centerline_segments_array[idx3][0][-1]
                    if idx3 not in [idx1, idx2]:
                        if np.linalg.norm(startpoint1 - startpoint3) < 1e-4 or np.linalg.norm(startpoint2 - endpoint3) < 1e-4:
                            is_bifurcation = True
                # If we have not been able to find a bifurcation, we proceed to make the final segment
                if not is_bifurcation:
                    # We first check which segment is floating (see if endpoint is shared with another segment)
                    endpoint1 = final_centerline_segments_array[idx1][0][-1]
                    endpoint2 = final_centerline_segments_array[idx2][0][-1]
                    for idx3 in range(len(final_centerline_segments_array)):
                        if np.linalg.norm(endpoint1 - final_centerline_segments_array[idx3][0][0]) < 1e-4 or np.linalg.norm(endpoint1 - final_centerline_segments_array[idx3][0][-1]) < 1e-4:
                            first_segment_idx = idx2
                            second_segment_idx = idx1
                            break
                        elif np.linalg.norm(endpoint2 - final_centerline_segments_array[idx3][0][0]) < 1e-4 or np.linalg.norm(endpoint2 - final_centerline_segments_array[idx3][0][-1]) < 1e-4:
                            first_segment_idx = idx1
                            second_segment_idx = idx2
                            break
                    # Now we can build the final segment, both with positions and radii
                    final_segment_positions = np.append(np.flip(final_centerline_segments_array[first_segment_idx][0], axis = 0), final_centerline_segments_array[second_segment_idx][0], axis = 0)
                    final_segment_radius = np.append(np.flip(final_centerline_segments_array[first_segment_idx][1], axis = 0), final_centerline_segments_array[second_segment_idx][1], axis = 0)
                    final_centerline_segments_array[first_segment_idx][0] = final_segment_positions
                    final_centerline_segments_array[first_segment_idx][1] = final_segment_radius
                    # We also keep the alternative index to delete it once the analysis is finished
                    delete_idx.append(second_segment_idx)

    # Finally, we delete the additional segments and save the array as a npy file
    final_centerline_segments_array = np.delete(final_centerline_segments_array, delete_idx, axis = 0)
    np.save(os.path.join(case_dir, "centerline_segments_array.npy"), final_centerline_segments_array)

def perform_segmentation_unification(case_dir):
    """
    Reads all segmentations in case_dir/segmentations 
    files and creates unified segmentation. 

    **** This needs re-thinking to be feasible, it takes forever as of now without the group id from the clipped models.
    
    Saves unified segmentation as:

    >>> case_dir/segmentation.vtk

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 

    Returns
    -------

    """
    print("Unifying all segmentations...")
    surface_model_list = sorted([surface_model_file for surface_model_file in os.listdir(os.path.join(case_dir, "segmentations")) if surface_model_file.endswith(".vtk")])
    # Initialize the vtkPoints and the vtkCellArray objects for the segmentation
    cell_array_segmentation = vtk.vtkCellArray()
    points_segmentation = vtk.vtkPoints()

    points_from_previous_segmentations = 0

    if len(surface_model_list) == 1:
        shutil.copyfile(os.path.join(case_dir, "segmentations", "segmentation0.vtk"), os.path.join(case_dir, "segmentation.vtk"))
    else:
        for surface_model_id, _ in enumerate(surface_model_list):
            # Load segmentation         
            segmentation_path = os.path.join(case_dir, "segmentations", "segmentation{}.vtk".format(surface_model_id))
            vtk_poly_data_reader = vtk.vtkPolyDataReader()
            vtk_poly_data_reader.SetFileName(segmentation_path)
            vtk_poly_data_reader.Update()
            segmentation = vtk_poly_data_reader.GetOutput()

            if segmentation.GetNumberOfCells() == 0:
                print("Error in segmentation {}. Skipping".format(surface_model_id))
            else:
                print("Processing segmentation {}...".format(surface_model_id))
                # To speed up computation, we bypass processing on the first segment
                # and we add the points and cells from the rest of the 
                if surface_model_id == 0:
                    for point_idx in range(segmentation.GetNumberOfPoints()):
                        points_segmentation.InsertNextPoint(segmentation.GetPoints().GetPoint(point_idx))
                    for cell_idx in range(segmentation.GetNumberOfCells()):
                        cell_array_segmentation.InsertNextCell(segmentation.GetCell(cell_idx))
                else:
                    for point_idx in range(segmentation.GetNumberOfPoints()):
                        points_segmentation.InsertNextPoint(segmentation.GetPoints().GetPoint(point_idx))
                    for cell_idx in range(segmentation.GetNumberOfCells()):
                        cell_array_segmentation.InsertNextCell(segmentation.GetCell(cell_idx))
                    # Get number of points in each segmentation cell (= 3)
                    number_of_point_ids = segmentation.GetCell(0).GetPointIds().GetNumberOfIds()
                    # # Generally, points are placed as cell indices go up, but this is not always the case
                    # # To speed up computations, we only search for cells with higher cellIds than the ones already searched for, 
                    # # But in the cases where a point_idx has not been found, we search across all cells of the model, in order
                    # # to ensure that no point_idx is missed
                    # last_cell = 0
                    # # We iterate through every pointId
                    # for point_idx in range(segmentation.GetNumberOfPoints()):
                    #     # We need a boolean variable to stop the iterative search when a point is found to speed up computations
                    #     found_point = False
                    #     # We primarily only search for cells with a cellId larger than the ones analyzed
                    #     # Limiting up the search dramatically speeds up computations
                    #     for cell_idx in range(max(0, last_cell - 1), segmentation.GetNumberOfCells()):
                    #         # Iterate over points in cell
                    #         for idx in range(number_of_point_ids):
                    #             # If a point is found with pointId equal to the next point_idx
                    #             if segmentation.GetCell(cell_idx).GetPointId(idx) == point_idx:
                    #                 # Keep cell_idx to limit cell of the next point_idx
                    #                 last_cell = cell_idx
                    #                 # Insert next point in final segmentation point object and groupId point array
                    #                 points_segmentation.InsertNextPoint(segmentation.GetCell(cell_idx).GetPoints().GetPoint(idx))
                    #                 # Update boolean marker to stop the search for the current pointidx
                    #                 found_point = True
                    #                 break
                    #         # Break cell serach if point is found
                    #         if found_point:
                    #             break
                    #     # If point is not found, search all throughout the cell pool, including cells with a smaller cell_idx than last_cell
                    #     # These searches are significantly longer than the general case, but we only apply them when needed
                    #     # This is very rare but if not done, it will mess up the final model
                    #     if not found_point:
                    #         # If point_idx has not been found, we also look at the previous cells (rare but it happens)
                    #         for cell_idx in range(segmentation.GetNumberOfCells()):
                    #             # Iterate over points in cell
                    #             for idx in range(number_of_point_ids):
                    #                 # If a point is found with pointId equal to the next point_idx
                    #                 if segmentation.GetCell(cell_idx).GetPointId(idx) == point_idx:
                    #                     # Keep cell_idx to limit cell of the next point_idx
                    #                     last_cell = cell_idx
                    #                     # Insert next point in final segmentation point object and groupId point array
                    #                     points_segmentation.InsertNextPoint(segmentation.GetCell(cell_idx).GetPoints().GetPoint(idx))
                    #                     # Update boolean marker to stop the search for the current pointidx
                    #                     found_point = True
                    #             # Break cell serach if point is found
                    #             if found_point:
                    #                 break

                    # We need this to set the new pointIds for the triangles with the SetId method. This will be 3
                    # Insert the cells with the corresponding groupId to the new vtkCellArray
                    # for idx in cellIdArray:
                    for cell_idx in range(segmentation.GetNumberOfCells()):
                        cell = vtk.vtkTriangle()
                        cell.GetPointIds().SetNumberOfIds(number_of_point_ids)
                        for idx in range(number_of_point_ids):
                            cell.GetPointIds().SetId(idx, segmentation.GetCell(cell_idx).GetPointId(idx) + points_from_previous_segmentations)
                        cell_array_segmentation.InsertNextCell(cell)
                
                # Update total number of points from previous models
                points_from_previous_segmentations += segmentation.GetNumberOfPoints()

        # Store all segmentation data in new vtkPolyData
        final_segmentation = vtk.vtkPolyData()
        final_segmentation.SetPoints(points_segmentation)
        final_segmentation.SetPolys(cell_array_segmentation)

        # We can to compute the normals for all mesh triangles
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputData(final_segmentation)
        normals.SetFeatureAngle(80)
        normals.AutoOrientNormalsOn()
        normals.UpdateInformation()
        normals.Update()
        final_segmentation = normals.GetOutput()

        # We also pass a clean vtkPolyData filter for good measure
        clean_poly_data = vtk.vtkCleanPolyData()
        clean_poly_data.SetInputData(final_segmentation)
        clean_poly_data.Update()
        final_segmentation = clean_poly_data.GetOutput()

        writer = vtk.vtkPolyDataWriter()
        writer.SetFileVersion(42)
        writer.SetInputData(final_segmentation)
        writer.SetFileName(os.path.join(case_dir, "segmentation.vtk"))
        writer.Write()