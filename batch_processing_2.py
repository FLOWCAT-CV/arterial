import os

database_dir = "/media/Disk_B/databases/arterial_not_processed"

skip = []

for case_id in sorted(os.listdir(database_dir))[:25]:
    if os.path.isfile(os.path.join(database_dir, case_id, "{}_segmentation.nii.gz".format(case_id))):
        case_dir = os.path.join(database_dir, case_id)
        print("Processing case {} (segmentation available)".format(case_id))
        # # os.system("python /home/vhir/github/arterial/perform_analysis.py -case_dir {}".format(case_dir))
        os.system("xvfb-run --auto-servernum --server-num=1 /home/vhir/anaconda3/envs/arterial_env/bin/python /home/vhir/github/arterial/perform_analysis.py -case_dir {} -ss t -sb t -sc t -sfe t".format(case_dir))