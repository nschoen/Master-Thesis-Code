import os
from os.path import join
import shutil

SRC = r"C:\Users\i00110578\OneDrive - Endress+Hauser\EH-Datasets\.prt Downloads nach Klassen\Adapter_73CF30"
DEST = r"C:\Users\i00110578\projects\local_datasets\eh_pro_adapters\obj"

for dirpath, _dirnames, filenames in os.walk(SRC):
    if len(filenames) > 0:
        document = dirpath.split("\\")[-2]
        version = dirpath.split("\\")[-1][2:]
    for filename in filenames:
        extension = filename.split('.')[-1]
        src = join(dirpath, filename)
        dest = join(DEST, f"{document}-{version}.{extension}")
        shutil.copyfile(src, dest)
