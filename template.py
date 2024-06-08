import os
from pathlib import Path
import logging

logging.basicConfig(format='[%(asctime)s]: %(message)s:', level=logging.INFO)

list_of_files = [
    'src/__init__.py',
    'src/helper.py',
    'src/prompt.py',
    '.env',
    'requirements.txt',
    'setup.py',
    'research/trials.ipynb',
    'app.py',
    'test.py'
]

for file in list_of_files:
    filepath = Path(file)
    filedir, filename = os.path.split(filepath)

    if filedir!= "":
        os.makedirs(filedir, exist_ok=True)

    if (not os.path.exists(filepath)) or (os.path.getsize(filepath) == 0):
        with open(filepath, "w") as f:
            pass
        logging.info(f"Creating empty file : {filepath}")
            # logging.info(f"Creating empty file : {filepath}")

    else:
        logging.info(f"{filename} already exists")