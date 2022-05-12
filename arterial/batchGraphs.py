import os
import shutil

databaseDir = "/Users/pere/opt/anaconda3/envs/arterialenv/Data/Arterial/Database"
externalHDPath = "/Volumes/My Passport/Arterial II/Database"

# skip = ["18545937"]
# skip = ["Revisit", "Unsolved"]
skip = []

for caseId in [x for x in sorted(os.listdir(databaseDir)) if not x.startswith(".") and x not in skip][:1]:

    # if not os.path.isfile(os.path.join(databaseDir, caseId, "supersegments.png")):
        casePath = os.path.join(databaseDir, caseId, f"{caseId}.nii.gz")

        if os.path.isfile(os.path.join(externalHDPath, caseId, f"{caseId}_CTA.nii.gz")):
            shutil.copyfile(os.path.join(externalHDPath, caseId, f"{caseId}_CTA.nii.gz"), os.path.join(databaseDir, caseId, f"{caseId}_CTA.nii.gz"))
        else:
            shutil.copyfile(os.path.join(externalHDPath, caseId, f"{caseId}.nii.gz"), os.path.join(databaseDir, caseId, f"{caseId}_CTA.nii.gz"))

        os.system(f"python3 /Users/pere/GitHub/arterial/arterial/performAnalysis.py -casePath {casePath}")

        os.remove(os.path.join(databaseDir, caseId, f"{caseId}_CTA.nii.gz"))

# for caseId in [x for x in sorted(os.listdir(externalHDPath)) if not x.startswith(".") and x not in skip]:
#     if len([x for x in sorted(os.listdir(os.path.join(externalHDPath, caseId))) if x.endswith(".nii.gz")]) < 2:
#         # print(caseId, [x for x in sorted(os.listdir(os.path.join(externalHDPath, caseId))) if x.endswith(".nii.gz")])
#         shutil.move(os.path.join(externalHDPath, caseId), os.path.join("/Volumes/My Passport/Arterial II/Fix", caseId))