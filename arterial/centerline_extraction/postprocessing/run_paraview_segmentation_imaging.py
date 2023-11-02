#   Copyright 2022 Stroke Research at Vall d'Hebron Research Institute (VHIR), Barcelona, Spain.

import os

def get_rotation_png_files(case_dir):
    """
    Wrapper function for the image acquisition of a 360 degree view
    of a case_dir/segmentation.vtk object. A PNG file will be generated
    every 10 degrees. Images will be stored in:

    >>> case_dir/segmentation_images/{idx}.png

    Where idx will be an ordered 5-digit number from 0 to 36.

    Parameters
    ----------
    case_dir : string or path-like object
        Path to case directory. 

    Returns
    -------

    """
    PARAVIEW_PATH = os.environ["paraview_path"]
    RUN_SEGMENTATION_ROTATION_PARAVIEW_SCRIPT = os.path.join(os.environ["arterial_dir"], "centerline_extraction/postprocessing/run_paraview_segmentation_imaging.py")

    os.system("{} {} -case_dir {}".format(PARAVIEW_PATH, RUN_SEGMENTATION_ROTATION_PARAVIEW_SCRIPT, case_dir))

if __name__ == "__main__":
    # Script to be executed by Paraview Python interpreter
    # Will only be executed when this script is called from a direct terminal command
    import os, argparse

    import vtk
    from paraview.simple import *

    import numpy as np

    from utils import set_default_display_properties

    parser = argparse.ArgumentParser()
    parser.add_argument("-case_dir", "--case_dir", type=str, required=True,
        help="Path to directory containing the analysis data. Required.")
        
    parser = parser.parse_args()

    case_dir = parser.case_dir

    rotation_step = 10

    # Disable automatic camera reset on 'Show'
    paraview.simple._DisableFirstRenderCameraReset()

    # Create a new 'Legacy VTK Reader'
    segmentation_vtk = LegacyVTKReader(registrationName='extracranial_vessels_segmentation.vtk', FileNames=[os.path.join(case_dir, "extracranial_vessels_segmentation.vtk")])

    # Set active source
    SetActiveSource(segmentation_vtk)

    # Get active view
    renderView1 = GetActiveViewOrCreate('RenderView')

    # Show data in view
    segmentation_vtk_display = Show(segmentation_vtk, renderView1, 'GeometryRepresentation')
    # Set default display properties
    segmentation_vtk_display = set_default_display_properties(segmentation_vtk_display)

    # Change solid color
    segmentation_vtk_display.AmbientColor = [1.0, 0.09411764705882353, 0.10980392156862745]
    segmentation_vtk_display.DiffuseColor = [1.0, 0.09411764705882353, 0.10980392156862745]

    LoadPalette(paletteName='BlackBackground')

    # Get layout
    layout1 = GetLayout()

    # Layout/tab size in pixels
    layout1.SetSize(414, 600)

    if not os.path.isdir(os.path.join(case_dir, "segmentation_images")): os.mkdir(os.path.join(case_dir, "segmentation_images"))

    # Get active camera
    camera = GetActiveCamera()

    # Set default transform
    default_transform = vtk.vtkTransform()
    default_transform.RotateZ(180)
    default_transform.RotateX(90)

    # Get bounds
    source = GetActiveSource()
    bounds = source.GetDataInformation().GetBounds()

    # Compute source center
    center = np.array([
        (bounds[1] + bounds[0]) / 2,
        (bounds[3] - bounds[2]) / 2,
        (bounds[5] - bounds[4]) / 2
    ])

    # Add to transform
    default_transform.Translate(-center)

    # Apply rotation and render
    camera.ApplyTransform(default_transform)

    # Reset view to fit data
    renderView1.ResetCamera(False)
    camera.SetViewAngle(20)
    Render()

    # Set new iterative rotation transform
    snap_rotation = vtk.vtkTransform()
    # Set new rotation for each iteration
    snap_rotation.RotateZ(rotation_step)

    idx = 0

    for angle in range(0, 361, rotation_step):

        identifier = str(idx)
        while len(identifier) < 5:
            identifier = "0" + identifier

        # save screenshot
        SaveScreenshot(os.path.join(case_dir, "segmentation_images", "{}.png".format(identifier)), renderView1, ImageResolution=[414, 600],
            FontScaling='Scale fonts proportionally',
            OverrideColorPalette='',
            StereoMode='No change',
            TransparentBackground=0, 
            # PNG options
            CompressionLevel='5',
            MetaData=['Application', 'ParaView'])

        # Apply small rotation
        camera.ApplyTransform(snap_rotation)
        # Reset view to fit data
        renderView1.ResetCamera(False)
        camera.SetViewAngle(20)
        Render()

        idx += 1