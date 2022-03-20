import open3d as o3d
import creopyson
import os
import shutil
import time
import csv
import math

c = creopyson.Client()
c.connect()

use_step_size = 0
target = 10000
target_ref = 7500
max_target = 1500

if not c.is_creo_running():
    raise Exception("Creo is not running")

MAP_KEY = "~ Trail `UI Desktop` `UI Desktop` `UIT_TRANSLUCENT` `NEED_TO_CLOSE`;\
~ Select `main_dlg_cur` `PHTLeft.AssyTree` 1 `node0`;\
~ Command `ProCmdModelSaveAs`;\
~ Activate `file_saveas` `opt_EMBED_BROWSER_TB_SAB_LAYOUT`;\
~ Open `file_saveas` `type_option`;~ Close `file_saveas` `type_option`;\
~ Select `file_saveas` `type_option` 1 `db_549`;\
~ Activate `file_saveas` `OK`;"

MAP_KEY_CHORD_HEIGHT = "~ Update `export_slice` `ChordHeightPanel` `{CHORD_HEIGHT}`;\
~ FocusOut `export_slice` `ChordHeightPanel`;"

MAP_KEY_END = "~ Activate `export_slice` `UseSSCheckButton` {use_step_size};\
~ Activate `export_slice` `OK`;"

def convert_part_to_step(file, destination_folder, chord_height):
    filename = os.path.basename(file)
    filename_base = filename.split('.')[0]
    dirname = os.path.abspath(os.path.dirname(file))

    lowercase_stl_filename = f"{filename_base.lower()}.stl"
    stl_filename = f"{filename_base}.stl"
    destination_file = os.path.join(destination_folder, stl_filename)

    c.creo_cd(dirname)
    time.sleep(1)
    c.file_open(f"{filename_base}.prt", dirname=dirname, display=True)

    mapkey_script = MAP_KEY.replace('\n', '').replace('{use_step_size}', f"{use_step_size}")
    mapkeys = mapkey_script.split(';')

    if chord_height:
        mapkey_scrip_ch = MAP_KEY_CHORD_HEIGHT.replace('\n', '').replace('{CHORD_HEIGHT}', f"{chord_height}")
        for mk in mapkey_scrip_ch.split(';'):
            mapkeys.append(mk)

    mapkey_scrip_end = MAP_KEY_END.replace('\n', '')
    for mk in mapkey_scrip_end.split(';'):
        mapkeys.append(mk)

    j = 0
    for mk in mapkeys:
        time.sleep(0.1)
        c.interface_mapkey(f"{mk};")
        j += 0

    from_path = os.path.join(dirname, lowercase_stl_filename)
    time.sleep(1.5)

    shutil.move(from_path, destination_file)

    time.sleep(0.2)
    c.file_close_window()  # important
    return destination_file

def count_edges(mesh):
    mesh.compute_adjacency_list()
    edge_count = 0
    # i = 0
    for i, al in enumerate(mesh.adjacency_list):
        for r in al:
            if i < r:
                edge_count += 1
    return edge_count

def convert_folder(source_folder, destination_folder):
    """Iterates over files in the specified folder and converts the .prt files to .stl files or
    recursively calls the function again if the child is a folder"""

    min_surface_area_to_edges = 10000000
    max_surface_area_to_edges = 0

    with open('stl_stats_default_toy9-stepsize.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(["num_edges", "num_vertices"," num_triangles"," is_manifold"," is_watertight"," surface_area",
                         "ratio_surface_to_edges"," ratios_triangles_to_edges"," mean_triangle_size"])

        for dirpath, _dirnames, filenames in os.walk(source_folder):
            for filename in filenames:
                if filename.split('.')[1].lower() != 'prt':
                    continue

                relpath = os.path.relpath(dirpath, source_folder)
                destination_path = os.path.join(destination_folder, relpath)
                if not os.path.exists(destination_path):
                    os.makedirs(destination_path, exist_ok=True)
                file = os.path.join(dirpath, filename)  # .replace("\\","/")

                basename = os.path.basename(file).replace('.prt', '.stl')
                stl_file = os.path.join(destination_path, basename)
                if os.path.isfile(stl_file):
                    mesh = o3d.io.read_triangle_mesh(stl_file)
                    num_triangles = len(mesh.triangles)
                    if num_triangles > target:
                        continue
                    else:
                        os.remove(stl_file)


                chord_height = None
                for i in range(15):
                    stl_file = convert_part_to_step(file, destination_path, chord_height)
                    mesh = o3d.io.read_triangle_mesh(stl_file)
                    #num_edges = count_edges(mesh)
                    num_triangles = len(mesh.triangles)
                    print("num_triangles", num_triangles)
                    if chord_height == None and num_triangles > target:
                        # chord height = None means this is the default chord height
                        # thus everything above target is allowed
                        break
                    elif target < num_triangles < max_target:
                        # chord height has been set programatically at iteration 1+
                        # make sure that the number of triangles i at least higher then target not to to high (max_target)
                        break

                    if chord_height == None:
                        # initial chord height value
                        chord_height = 0.5
                    else:
                        factor = target_ref / num_triangles
                        chord_height = chord_height / factor

                    print("chord_height", chord_height)

                    os.remove(stl_file)

                num_edges = count_edges(mesh)
                num_vertices = len(mesh.vertices)
                num_triangles = len(mesh.triangles)
                is_manifold = mesh.is_edge_manifold()
                is_watertight = mesh.is_watertight()
                surface_area = mesh.get_surface_area()
                ratio_surface_to_edges = surface_area / num_edges
                ratios_triangles_to_edges = num_triangles / num_edges
                mean_triangle_size = surface_area / num_triangles
                writer.writerow([num_edges, num_vertices, num_triangles, is_manifold, is_watertight, surface_area,
                                 ratio_surface_to_edges, ratios_triangles_to_edges, mean_triangle_size])

                print("num_edges", num_edges)
                print("num_triangles", num_triangles)
                print("is_manifold", is_manifold)
                print("ratio_surface_to_edges", ratio_surface_to_edges)
                print("mean_triangle_size", mean_triangle_size)

                if ratio_surface_to_edges < min_surface_area_to_edges:
                    min_surface_area_to_edges = ratio_surface_to_edges

                if max_surface_area_to_edges > min_surface_area_to_edges:
                    max_surface_area_to_edges = ratio_surface_to_edges

    print("min_surface_area_to_edges", min_surface_area_to_edges)
    print("max_surface_area_to_edges", max_surface_area_to_edges)

if __name__ == '__main__':
    convert_folder("C:\\Users\\i00110578\\projects\\AIAx-Use-Case-1\\datasets\\MMM\\toy9-to-stl-creo\\flange-custom",
                   "C:\\Users\\i00110578\\projects\\AIAx-Use-Case-1\\datasets\\MMM\\toy9-to-stl-creo\\stl-min-9000-triangles")
    #convert_folder("C:\\Users\\i00110578\\projects\\AIAx-Use-Case-1\\datasets\\EH\\part_to_stl_creo\\prt",
    #              "C:\\Users\\i00110578\\projects\\AIAx-Use-Case-1\\datasets\\EH\\part_to_stl_creo\\stl-min-1000-triangles")