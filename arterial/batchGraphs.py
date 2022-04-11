import os

databaseDir = "/Users/pere/opt/anaconda3/envs/gnnenv/graphUNet/denseGraphs/database/testDatabase"

skip = ["12878665"]
# skip = ["12701255", "12721439", "13073698", "13791631"]

for caseId in [x for x in sorted(os.listdir(databaseDir)) if not x.startswith(".") and x not in skip]:

    casePath = os.path.join(databaseDir, caseId, f"{caseId}.nii.gz")

    try:
        os.remove(os.path.join(databaseDir, caseId, "supersegmentsLabel.png"))
        # os.rename(os.path.join(databaseDir, caseId, "supersegmentsLabelOLD.png"), os.path.join(databaseDir, caseId, "supersegmentsLabelv0.png"))
        # os.rename(os.path.join(databaseDir, caseId, "supersegmentsLabel.png"), os.path.join(databaseDir, caseId, "supersegmentsLabelv1.png"))
    except:
        pass
    try:
        os.remove(os.path.join(databaseDir, caseId, "supersegmentsPred.png"))
        # os.rename(os.path.join(databaseDir, caseId, "supersegmentsPredOLD.png"), os.path.join(databaseDir, caseId, "supersegmentsPredv0.png"))
        # os.rename(os.path.join(databaseDir, caseId, "supersegmentsPred.png"), os.path.join(databaseDir, caseId, "supersegmentsPredv1.png"))
    except:
        pass

    os.system(f"python3 /Users/pere/GitHub/arterial/arterial/performAnalysis.py -casePath {casePath}")

    try:
        os.rename(os.path.join(databaseDir, caseId, "supersegmentsLabel.png"), os.path.join(databaseDir, caseId, "supersegmentsLabelv2.png"))
    except:
        pass   
    try:
        os.rename(os.path.join(databaseDir, caseId, "supersegmentsPred.png"), os.path.join(databaseDir, caseId, "supersegmentsPredv2.png"))
    except:
        pass
