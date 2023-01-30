import os

database_dir = "/media/Disk_B/databases/arterial_not_processed"

skip = ["11231061"]

for case_id in sorted(os.listdir(database_dir))[25:50]:
    if case_id not in skip:
        case_dir = os.path.join(database_dir, case_id)
        print("Processing case {}".format(case_id))
        # os.system("python /home/vhir/github/arterial/perform_analysis.py -case_dir {}".format(case_dir))
        os.system("/home/vhir/anaconda3/envs/arterial_env/bin/python /home/vhir/github/arterial/perform_analysis.py -case_dir {} -ss t -sc t".format(case_dir))