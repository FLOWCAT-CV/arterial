import os
import vtk
import shutil
from time import time

databaseDir = "/Users/pere/opt/anaconda3/envs/gnnenv/gnn_arterial/database/DatasetSize102_lowres_SGD_ReduceOnPlateau_713_fold_1/train"
auxPath = "/Users/pere/opt/anaconda3/envs/gnnenv/gnn_arterial/data/ManualMeasurements"

skip = ["fix", "fix clipped model"]

for caseDir in [x for x in sorted(os.listdir(databaseDir)) if not x.startswith(".") and x not in skip]:

    # casePath = os.path.join(databaseDir, caseDir, f"{caseDir}.nii.gz")

    # os.system(f"python /Users/pere/GitHub/arterial/v2/arterial/performAnalysis.py -casePath {casePath}")

    # os.mkdir(os.path.join(databaseDir, caseDir[:-7]))
    # os.rename(os.path.join(databaseDir, caseDir), os.path.join(databaseDir, caseDir[:-7], caseDir))

    shutil.copyfile(os.path.join(databaseDir, caseDir, "graph.pickle"), os.path.join(auxPath, "graphs", f"{caseDir}.pickle"))
    shutil.copyfile(os.path.join(databaseDir, caseDir, "graph_label.pickle"), os.path.join(auxPath, "labels", f"{caseDir}.pickle"))