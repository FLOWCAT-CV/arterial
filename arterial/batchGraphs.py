import os

databaseDir = "/Users/pere/opt/anaconda3/envs/gnnenv/graphUNet/denseGraphs/database/testDatabase"

skip = ["12878665"]
# skip = ["16871012"]

for caseId in [x for x in sorted(os.listdir(databaseDir)) if not x.startswith(".") and x not in skip]:

    casePath = os.path.join(databaseDir, caseId, f"{caseId}.nii.gz")

    # os.system(f"python3 /Users/pere/GitHub/dev/arterial/arterial/performAnalysis.py -casePath {casePath}")

    try:
        os.remove(os.path.join(databaseDir, caseId, "supersegmentsPredv0.png"))
        os.remove(os.path.join(databaseDir, caseId, "supersegmentsPredv1.png"))
        os.rename(os.path.join(databaseDir, caseId, "supersegmentsPredv2.png"), os.path.join(databaseDir, caseId, "supersegmentsPred.png"))
        os.remove(os.path.join(databaseDir, caseId, "supersegmentsLabelv0.png"))
        os.remove(os.path.join(databaseDir, caseId, "supersegmentsLabelv1.png"))
        os.rename(os.path.join(databaseDir, caseId, "supersegmentsLabelv2.png"), os.path.join(databaseDir, caseId, "supersegmentsLabel.png"))
    except:
        print(caseId)
