import open3d as o3d
import creopyson
import os
import shutil
import time
import csv
import math
import random


c = creopyson.Client()
c.connect()

if not c.is_creo_running():
    raise Exception("Creo is not running")


#MAP_KEY = "~ Command `ProCmdModelSaveAs` ;~ Update `file_saveas` `opt_EMBED_BROWSER_TB_SAB_LAYOUT` `{DESTINATION_PATH}`;~ Open `file_saveas` `type_option`;~ Close `file_saveas` `type_option`;~ Select `file_saveas` `type_option` 1 `db_549`;~ Activate `file_saveas` `OK`;~ Update `export_slice` `ChordHeightPanel` `0.5`;~ Update `export_slice` `AngleControlPanel` `0.5`;~ Update `export_slice` `StepSizePanel` `{STEP_SIZE}`;~ Activate `export_slice` `OK`;"
# ~ LButtonArm `file_saveas` `tb_EMBED_BROWSER_TB_SAB_LAYOUT` 3 431 14 0;\
# ~ LButtonDisarm `file_saveas` `tb_EMBED_BROWSER_TB_SAB_LAYOUT` 3 431 14 0;\
# ~ LButtonActivate `file_saveas` `tb_EMBED_BROWSER_TB_SAB_LAYOUT` 3 431 14 0;\
MAP_KEY = "~ Trail `UI Desktop` `UI Desktop` `UIT_TRANSLUCENT` `NEED_TO_CLOSE`;~ Select `main_dlg_cur` `PHTLeft.AssyTree` 1 `node0`;~ Command `ProCmdModelSaveAs`;\
~ Activate `file_saveas` `opt_EMBED_BROWSER_TB_SAB_LAYOUT`;\
~ Open `file_saveas` `type_option`;~ Close `file_saveas` `type_option`;\
~ Select `file_saveas` `type_option` 1 `db_549`;\
~ Activate `file_saveas` `OK`;\
~ Update `export_slice` `ChordHeightPanel` `{CHORD_HEIGHT}`;\
~ FocusOut `export_slice` `ChordHeightPanel`;\
~ Update `export_slice` `AngleControlPanel` `{AngleControlPanel}`;\
~ FocusOut `export_slice` `AngleControlPanel`;\
~ Activate `export_slice` `UseSSCheckButton` 1;\
~ Update `export_slice` `StepSizePanel` `{STEP_SIZE}`;\
~ FocusOut `export_slice` `StepSizePanel`;~ Activate `export_slice` `OK`;"


# target_edge_count = 24000
# min_edge_count = 20000
# max_edge_count = 28000
# target_edge_count = 3400
# min_edge_count = 2800
# max_edge_count = 4000

# target_edge_count = 9000
# min_edge_count = 8500
# min_edge_count_2 = 7500
# max_edge_count = 10000
# max_edge_count_2 = 12500

#target_edge_count = 32000
# min_edge_count = 30000
# min_edge_count_2 = 28000
# max_edge_count = 36000
# max_edge_count_2 = 40000

target_edge_count = 9000
min_edge_count = 8500
min_edge_count_2 = 8000
max_edge_count = 9500
max_edge_count_2 = 10000

def convert_part_to_step(file, destination_folder, step_size=1.0, set_chord_height=None, i=0, loop=True):
    # {STEP_SIZE}
    # {DESTINATION_PATH}

    filename = os.path.basename(file)
    filename_base = filename.split('.')[0]
    dirname = os.path.abspath(os.path.dirname(file))

    lowercase_stl_filename = f"{filename_base.lower()}.stl"
    stl_filename = f"{filename_base}.stl"
    destination_file = os.path.join(destination_folder, stl_filename)

    c.creo_cd(dirname)
    # if not RESET_DESTINATION_FOLDER:
    #     if os.path.exists(os.path.join(destination_folder, f"{filename_base}.stp")):
    #         print(f"{filename_base}.step exists already, continue")
    #         return
    time.sleep(1)
    c.file_open(f"{filename_base}.prt", dirname=dirname, display=True)

    #mapkey_script = MAP_KEY.replace('\n', '').replace("{DESTINATION_PATH}", destination_folder).replace("{STEP_SIZE}", "5.0")
    chord_height = step_size * 0.1
    #chord_height = step_size * 0.1
    #if set_chord_height:
    #    chord_height = set_chord_height
    angle_control_panel = 0.25
    #angle_control_panel = 0.5
    if i > 14:
        #angle_control_panel = 0.1
        angle_control_panel = step_size / 0.1

    #mapkey_script = MAP_KEY.replace("{STEP_SIZE}", f"{step_size}").replace("{CHORD_HEIGHT}", f"{chord_height}").replace("{AngleControlPanel}", f"{angle_control_panel}")
    mapkey_script = MAP_KEY.replace('\n', '').replace("{STEP_SIZE}", f"{step_size}").replace("{CHORD_HEIGHT}", f"{chord_height}").replace("{AngleControlPanel}", f"{angle_control_panel}")
    #print("mapkey_script", mapkey_script)
    mapkeys = mapkey_script.split(';')
    #print("mapkeys", mapkeys)
    j = 0
    for mk in mapkeys:
        if j > 5:
            time.sleep(0.2)
        else:
            time.sleep(0.1)
        c.interface_mapkey(f"{mk};")
        j += 0

    waited = 0
    from_path = os.path.join(dirname, lowercase_stl_filename)
    time.sleep(1.5)
    i = 0
    while not os.path.exists(from_path) and loop:
        print(i, "versuch")
        step_size = step_size * random.uniform(0.9, 1.1)
        convert_part_to_step(file, destination_folder, step_size=step_size, set_chord_height=set_chord_height, i=i, loop=False)
        time.sleep(2)
        if i > 5:
            raise Exception("nope")
        i += 1

    if not loop:
        return

    # while not os.path.exists(from_path) and waited < 20:
    #     time.sleep(2.0)
    #     waited += 2
    #     # if waited == 10:
    #     #     for mk in mapkeys:
    #     #         time.sleep(0.2)
    #     #         c.interface_mapkey(f"{mk};")
    #print("destination_file", destination_file)
    shutil.move(from_path, destination_file)

    # export STEP file
    # c.interface_export_file(
    #     'STEP',
    #     filename=f"{filename_base}.stp",
    #     dirname=destination_folder,
    #     advanced=True
    # )

    #c.map
    time.sleep(0.2)
    c.file_close_window()  # important
    return destination_file


def count_edges(mesh):
    mesh.compute_adjacency_list()
    #edge_list = []
    edge_count = 0
    # i = 0
    for i, al in enumerate(mesh.adjacency_list):
        for r in al:
            if i < r:
                edge_count += 1
            # idx = f"{min(i, r)}-{max(i, r)}"
            # if idx not in edge_list:
            #     edge_list.append(idx)
        # i += 1
    return edge_count
    #return len(edge_list)


def convert_folder(source_folder, destination_folder):
    """Iterates over files in the specified folder and converts the .prt files to .stl files or
    recursively calls the function again if the child is a folder"""

    # delete step folder first
    # if RESET_DESTINATION_FOLDER:
    #     shutil.rmtree(destination_folder, ignore_errors=True, onerror=None)

    min_surface_area_to_edges = 10000000
    max_surface_area_to_edges = 0

    with open('stl_stats.csv', 'w', newline='') as csvfile:
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
                    num_edges = count_edges(mesh)
                    if min_edge_count_2 <= num_edges <= max_edge_count_2:
                        continue
                    else:
                        os.remove(stl_file)

                print("filename", filename)

                step_size = 2.0
                i = 0

                while True:
                    i += 1
                    stl_file = convert_part_to_step(file, destination_path, step_size=step_size, i=i)
                    mesh = o3d.io.read_triangle_mesh(stl_file)
                    num_edges = count_edges(mesh)

                    if i > 30:
                        raise Exception("This took to long - step size:", step_size)

                    if min_edge_count <= num_edges <= max_edge_count:
                        break

                    if i > 10 and min_edge_count_2 <= num_edges <= max_edge_count_2:
                        break

                    factor = math.sqrt(target_edge_count / num_edges)

                    if i <= 2:
                        factor = factor**1.7

                    if 2 < i <= 4:
                        factor = factor**1.3

                    if i > 10:
                        factor = math.sqrt(factor)
                    step_size = step_size / factor
                    os.remove(stl_file)

                    print("num_edges", num_edges)
                    print("factor", factor)
                    print("step_size", step_size)
                    continue

                    # if num_edges > max_edge_count:
                    #     factor = target_edge_count / num_edges
                    #     step_size = step_size / factor
                    #     os.remove(stl_file)
                    #     continue

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
    # convert_part_to_step("C:\\Users\\i00110578\\projects\\AIAx-Use-Case-1\\datasets\\EH\\final_ehp_prt\\70CE00_housing\\230004734-C.prt",
    #                      "C:\\Users\\i00110578\\Desktop\\tmp_trash\\export_map_key_test")

    #convert_folder("C:\\Users\\i00110578\\projects\\AIAx-Use-Case-1\\datasets\\EH\\part_to_stl_creo\\prt",
    #               "C:\\Users\\i00110578\\projects\\AIAx-Use-Case-1\\datasets\\EH\\part_to_stl_creo\\stl-32000")

    convert_folder("C:\\Users\\i00110578\\projects\\AIAx-Use-Case-1\\datasets\\MMM\\toy9-to-stl-creo\\flange-custom",
                   "C:\\Users\\i00110578\\projects\\AIAx-Use-Case-1\\datasets\\MMM\\toy9-to-stl-creo\\stl-9000")