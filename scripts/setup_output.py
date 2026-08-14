import datetime
import os
import sys
import uuid
import subprocess

def setup_output_directory(dirname: str = None, subdir_name: str = None):

    if dirname is None:
        if subdir_name:
            dirname = os.path.join("output", f"{subdir_name}-{uuid.uuid4()}")
        else:
            dirname = os.path.join("output", f"output-{uuid.uuid4()}")

    if subdir_name is not None:
        dirname = os.path.join(dirname, subdir_name)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    with open(os.path.join(dirname, 'info.txt'), 'w') as description_fout:
        git_hash = get_git_revision_hash()
        datetimestr = str(datetime.datetime.now())
        description_fout.write(f"Date: {datetimestr}\n")
        description_fout.write(f"Commit {git_hash}\n")
        command = " ".join(sys.argv)
        description_fout.write(f"Command: {command}\n")
    return dirname


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()




