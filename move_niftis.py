import os, shutil

source_dir = "/media/Disk_B/databases/visiocat_registry/database"
destination_dir = "/media/Disk_B/databases/arterial_not_proecssed"

for identifier in sorted([identifier for identifier in os.listdir(source_dir) if identifier.startswith("VC_HUVH")]):
    if not os.path.isdir(os.path.join(destination_dir, identifier)):
        os.mkdir(os.path.join(destination_dir, identifier))
        for pat in sorted(os.listdir(os.path.join(source_dir, identifier, "NIFTI"))):
            for stu in sorted([stu for stu in os.listdir(os.path.join(source_dir, identifier, "NIFTI", pat)) if "BASAL_0" in stu]):
                