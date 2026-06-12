#Deletes All Empty Folders

import os
import sys
from pathlib import Path

path_to_folder = 'C:/Users/fredy/Downloads/Coding_Projects/Kpop_idol_training_pipeline/output'
folder_path = Path(path_to_folder)


for f in folder_path.iterdir():
    if f.is_dir():
        newer_path = f
        files = [f.name for f in newer_path.iterdir()]
        if not files:
            os.rmdir(newer_path)

