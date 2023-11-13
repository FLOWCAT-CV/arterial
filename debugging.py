# import os
# from arterial.feature_extraction.local_features.feature_extraction import perform_local_feature_extraction
# from arterial.io.load_and_save_operations import load_pickle

# nhc = "10068570"
# db = "/media/Disk_B/databases/arterial/database"

# case_dir = os.path.join(db, nhc)

# if os.path.exists(os.path.join(case_dir, "dense_graph.pickle")):
#     centerline_graph = load_pickle(os.path.join(case_dir, "dense_graph.pickle"))
#     perform_local_feature_extraction(case_dir, centerline_graph)
# else:
#     print("No graph found for case %s" % nhc)
#     for file in os.listdir(case_dir):
#         print(file)

################################################################################

import os
import argparse
from arterial.run.processor import ArterialProcessor
# from pyvirtualdisplay import Display

nhc = "10538473"
db = "/media/Disk_B/databases/arterial/database"

# display = Display(visible=0, size=(1400, 900))
# display.start()
args = argparse.ArgumentParser()
args.case_dir = os.path.join(db, nhc)
args.mode = "extracranial_vessels"
args.no_display = False
args.skip_segmentation = True
args.fast_segmentation = True
args.skip_centerline_extraction = True
args.skip_branching = True
args.skip_clipping = True
args.skip_vessel_labelling = True
args.skip_feature_extraction = False
processor = ArterialProcessor(args)
times = processor.perform_analysis()
# display.stop()