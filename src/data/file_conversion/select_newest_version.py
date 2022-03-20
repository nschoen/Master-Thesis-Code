import os
from os.path import join
import shutil

SRC = r"C:\Users\i00110578\projects\local_datasets\eh_pro_adapters\stl"
DEST = r"C:\Users\i00110578\projects\local_datasets\eh_pro_adapters\stl-latest-version"

copy_files = set()
versions = {}

for dirpath, _dirnames, filenames in os.walk(SRC):
    for filename in filenames:
        document = filename.split('.')[0].split('-')[0]
        version = filename.split('.')[0].split('-')[1]
        if version == '':
            version = '-'
        if document not in copy_files:
            copy_files.add(document)
        if document in versions:
            if version > versions[document]:
                versions[document] = version
        else:
            versions[document] = version

print("copy_files", copy_files)
for document in copy_files:
    version = versions[document]
    name = f"{document}-{version}.stl"
    src = join(SRC, name)
    dest = join(DEST, name)
    shutil.copyfile(src, dest)
