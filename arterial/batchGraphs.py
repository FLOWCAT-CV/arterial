import os
import shutil

import json
import networkx as nx

databaseDir = "/Users/pere/opt/anaconda3/envs/arterialenv/Data/Arterial/Database"
externalHDPath = "/Volumes/My Passport/Arterial II/Database"

# skip = ["10068570", "12059501", "16771098"]
skip = []

for caseId in [x for x in sorted(os.listdir(databaseDir)) if not x.startswith(".") and x not in skip]:
    casePath = os.path.join(databaseDir, caseId, f"{caseId}.nii.gz")

    with open(os.path.join(databaseDir, caseId, "patientConfiguration.json")) as jsonFile:
        patientConfiguration = json.load(jsonFile)[caseId]

    # if not os.path.isfile(os.path.join("/Users/pere/opt/anaconda3/envs/arterialenv/Data/Arterial/graphDatabase", f"{caseId}.pickle")) and not os.path.isfile(os.path.join("/Users/pere/opt/anaconda3/envs/arterialenv/Data/Arterial/incorrectSupersegments", f"{caseId}.pickle")) and patientConfiguration["Laterality"] not in ["Bilateral", "Indeterminat"]:
    if not os.path.isfile(os.path.join("/Users/pere/opt/anaconda3/envs/arterialenv/Data/Arterial/onlyNodeFeaturesGraphDatabase", f"{caseId}.pickle")) and not os.path.isfile(os.path.join("/Users/pere/opt/anaconda3/envs/arterialenv/Data/Arterial/incorrectSupersegments", f"{caseId}.pickle")) and patientConfiguration["Laterality"] not in ["Bilateral", "Indeterminat"]:

        if os.path.isfile(os.path.join(externalHDPath, caseId, f"{caseId}_CTA.nii.gz")):
            shutil.copyfile(os.path.join(externalHDPath, caseId, f"{caseId}_CTA.nii.gz"), os.path.join(databaseDir, caseId, f"{caseId}_CTA.nii.gz"))
        else:
            shutil.copyfile(os.path.join(externalHDPath, caseId, f"{caseId}.nii.gz"), os.path.join(databaseDir, caseId, f"{caseId}_CTA.nii.gz"))

        os.system(f"python3 /Users/pere/GitHub/arterial/arterial/performAnalysis.py -casePath {casePath}")

        try:
            os.remove(os.path.join(databaseDir, caseId, f"{caseId}_CTA.nii.gz"))
        except:
            pass

    # if os.path.isdir(os.path.join(databaseDir, caseId, "thrombectomyConfiguration")):

    #     supersegment = nx.read_gpickle(os.path.join(databaseDir, caseId, "thrombectomyConfiguration", "supersegment.pickle"))

    #     with open(os.path.join(databaseDir, caseId, "patientConfiguration.json")) as jsonFile:
    #         patientConfiguration = json.load(jsonFile)[caseId]

    #     supersegment.graph["Time to first series"] = patientConfiguration["Time first angiography"]

    #     nx.write_gpickle(supersegment, os.path.join(databaseDir, caseId, "thrombectomyConfiguration", "supersegment.pickle"), protocol = 4)
    #     nx.write_gpickle(supersegment, os.path.join("/Users/pere/opt/anaconda3/envs/arterialenv/Data/Arterial/graphDatabase", f"{caseId}.pickle"), protocol = 4)


# Use if downloaded predictions have to be copied to all database sites in computer and external HD
# downloads = "/Users/pere/Downloads"
# localDatabase = "/Users/pere/Documents/Projects/Vall d'Hebron/Tortuositat/Database/Arterial II/Group 2"

# for caseId in [x[:-7] for x in sorted(os.listdir(downloads)) if not x.startswith(".") and x not in skip]:
    # if os.path.isfile(os.path.join(downloads, f"{caseId}.nii.gz")):
        # shutil.copyfile(os.path.join(downloads, f"{caseId}.nii.gz"), os.path.join(databaseDir, caseId, f"{caseId}.nii.gz"))
        # shutil.copyfile(os.path.join(downloads, f"{caseId}.nii.gz"), os.path.join(externalHDPath, caseId, f"{caseId}.nii.gz"))
        # shutil.copyfile(os.path.join(downloads, f"{caseId}.nii.gz"), os.path.join(localDatabase, caseId, f"{caseId}.nii.gz"))
        
        # Use if CTAs have to be transferred within external HD
        # if os.path.isfile(os.path.join(externalHDPath, "Proces", "CTA niftis", caseId, f"{caseId}.nii.gz")):
        #     print("Proces", caseId)
        #     shutil.copyfile(os.path.join(externalHDPath, "Proces", "CTA niftis", caseId, f"{caseId}.nii.gz"), os.path.join(externalHDPath, caseId, f"{caseId}_CTA.nii.gz"))
        # elif os.path.isfile(os.path.join(externalHDPath, "Arterial", "Database", caseId, f"{caseId}_CTA.nii.gz")):
        #     print("Proces", caseId)
        #     shutil.copyfile(os.path.join(externalHDPath, "Arterial", "Database", caseId, f"{caseId}_CTA.nii.gz"), os.path.join(externalHDPath, caseId, f"{caseId}_CTA.nii.gz"))