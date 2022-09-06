import os
import shutil

import json
import networkx as nx

# databaseDir = "/Users/pere/dev/Arterial/reviewPaper1/data/ManualMeasurements"
databaseDir = "/Users/pere/dev/Arterial/5_Time_to_first_series_prediction/graphFeaturization/Database"
# graphDatabaseDir = "/Users/pere/opt/anaconda3/envs/gnnenv/graphUNet/simpleGraphs/data/Arterial simple redesign"

# skip = ["11887522", "11987412", "12270290", "18747058", # graph processing error
#         "10250923", "11884066", "11905410", "12604929", "14126212", "14968691"] # centerline processing error
# skip = ["11564171", "11594574", "11775980", "76457"]
skip = ["10250923"]

for caseId in [x for x in sorted(os.listdir(databaseDir)) if not x.startswith(".") and x in skip]:
    casePath = os.path.join(databaseDir, caseId, f"{caseId}.nii.gz")

    # if not os.path.isfile(os.path.join(databaseDir, caseId, "supersegments.png")) and
    # if os.path.isfile(os.path.join(databaseDir, caseId, "graph_label.pickle")):
        # for filename in ["simpleGraph.pickle", "simpleGraph.png"]:
        #     if os.path.isfile(os.path.join(databaseDir, caseId, filename)):
        #         os.remove(os.path.join(databaseDir, caseId, filename))

    os.system(f"python3 /Users/pere/dev/Arterial/5_Time_to_first_series_prediction/graphFeaturization/arterial/performAnalysis.py -casePath {casePath}")

        # if os.path.isfile(os.path.join(databaseDir, caseId,"graph_label.pickle")) and os.path.isfile(os.path.join(databaseDir, caseId,"simpleGraph.pickle")):
        #     label = nx.read_gpickle(os.path.join(databaseDir, caseId, "graph_label.pickle"))
        #     graph = nx.read_gpickle(os.path.join(databaseDir, caseId, "simpleGraph.pickle"))

        #     for src, dst in label.edges:
        #         graph[src][dst]["Vessel type"] = label[src][dst]["Vessel type"]
        #         graph[src][dst]["Vessel type name"] = label[src][dst]["Vessel type name"]

        #     nx.write_gpickle(graph, os.path.join(databaseDir, caseId, "graph_label.pickle"))

        # try:
        #     shutil.copyfile(os.path.join(databaseDir, caseId, "graph_label.pickle"), os.path.join(graphDatabaseDir, "labels", caseId + ".pickle"))
        #     shutil.copyfile(os.path.join(databaseDir, caseId, "graph_label.png"), os.path.join(graphDatabaseDir, "images", caseId + ".png"))
        # except:
        #     pass