import os
import shutil

import json
import networkx as nx

databaseDir = "/Users/pere/opt/anaconda3/envs/arterialenv/Data/Arterial/Database"
externalHDPath = "/Volumes/My Passport/Arterial II/Database"

# skip = ["18545937"]
# skip = ["Revisit", "Unsolved"]
skip = []

for caseId in [x for x in sorted(os.listdir(databaseDir)) if not x.startswith(".") and x not in skip]:

    # if not os.path.isfile(os.path.join(databaseDir, caseId, "supersegments.png")):
        casePath = os.path.join(databaseDir, caseId, f"{caseId}.nii.gz")

        # if os.path.isfile(os.path.join(externalHDPath, caseId, f"{caseId}_CTA.nii.gz")):
        #     shutil.copyfile(os.path.join(externalHDPath, caseId, f"{caseId}_CTA.nii.gz"), os.path.join(databaseDir, caseId, f"{caseId}_CTA.nii.gz"))
        # else:
        #     shutil.copyfile(os.path.join(externalHDPath, caseId, f"{caseId}.nii.gz"), os.path.join(databaseDir, caseId, f"{caseId}_CTA.nii.gz"))

        # os.system(f"python3 /Users/pere/GitHub/arterial/arterial/performAnalysis.py -casePath {casePath}")

        # os.remove(os.path.join(databaseDir, caseId, f"{caseId}_CTA.nii.gz"))

        if os.path.isdir(os.path.join(databaseDir, caseId, "thrombectomyConfiguration")):

            supersegment = nx.read_gpickle(os.path.join(databaseDir, caseId, "thrombectomyConfiguration", "supersegment.pickle"))

            with open(os.path.join(databaseDir, caseId, "patientConfiguration.json")) as jsonFile:
                patientConfiguration = json.load(jsonFile)[caseId]

            supersegment.graph["Time to first series"] = patientConfiguration["Time first angiography"]

            nx.write_gpickle(supersegment, os.path.join(databaseDir, caseId, "thrombectomyConfiguration", "supersegment.pickle"), protocol = 4)

# for caseId in [x for x in sorted(os.listdir(externalHDPath)) if not x.startswith(".") and x not in skip]:
#     if len([x for x in sorted(os.listdir(os.path.join(externalHDPath, caseId))) if x.endswith(".nii.gz")]) < 2:
#         # print(caseId, [x for x in sorted(os.listdir(os.path.join(externalHDPath, caseId))) if x.endswith(".nii.gz")])
#         shutil.move(os.path.join(externalHDPath, caseId), os.path.join("/Volumes/My Passport/Arterial II/Fix", caseId))