# The following code only works when the python interpreter is the same used to build the installed FreeCAD version
# instead just copy the code in the next block into the FreeCAD console

# FREECADPATH = 'C:/Program Files/FreeCAD/lib/' # path to your FreeCAD.so or FreeCAD.dll file
# import sys
# sys.path.append(FREECADPATH)

# Copy the code in three parts into the FreeCAD console, otherwise the parse will not understand it
import Part
import Mesh
import os

SOURCE_FOLDER = r"..\..\..\..\projects\local_datasets\toy3\extras\train-stp"
DESTINATION_FOLDER = r"..\..\..\..\projects\local_datasets\toy3\extras\train-stl"

doc = App.newDocument('Doc')
pf = doc.addObject("Part::Feature", "MyShape")

# first copy&paste into FreeCAD console untill here

def convert_step_to_stl(file, destination_folder):
    filename = os.path.basename(file)
    filename_base = filename.split('.')[0]
    shape = Part.Shape()
    shape.read(file)
    pf.Shape = shape
    print("Export file", filename_base)
    destination_file = os.path.join(destination_folder, f"{filename_base}.stl")
    Mesh.export([pf], destination_file)


# second copy&paste into FreeCAD console untill here (only the function), then the rest

for dirpath, _dirnames, filenames in os.walk(SOURCE_FOLDER):
    for filename in filenames:
        print("os.path.basename(filename)[1]", filename.split('.')[1])
        if filename.split('.')[1] != 'stp':
            continue
        relpath = os.path.relpath(dirpath, SOURCE_FOLDER)
        destination_path = os.path.join(DESTINATION_FOLDER, relpath)
        if os.path.exists(os.path.join(destination_path, f"{filename.split('.')[0]}.stl")):
            continue
        if not os.path.exists(destination_path):
            os.makedirs(destination_path, exist_ok=True)
        file = os.path.join(dirpath, filename)
        convert_step_to_stl(file, destination_path)