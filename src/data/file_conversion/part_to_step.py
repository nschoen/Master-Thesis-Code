import creopyson
import os
import shutil

# specifies if the destination folder should be deleted on each start or if
# the script should keep it and ignore existing .step files
RESET_DESTINATION_FOLDER = False

# Before running the code, start creo and creoson.
c = creopyson.Client()


def connect_creo_api():
    c.connect()

    return c.is_creo_running()


def convert_part_to_step(file, destination_folder):
    filename = os.path.basename(file)
    filename_base = filename.split('.')[0]
    dirname = os.path.dirname(file)

    if not RESET_DESTINATION_FOLDER:
        if os.path.exists(os.path.join(destination_folder, f"{filename_base}.stp")):
            print(f"{filename_base}.step exists already, continue")
            return

    c.file_open(f"{filename_base}.prt", dirname=dirname, display=True)

    # export STEP file
    c.interface_export_file(
        'STEP',
        filename=f"{filename_base}.stp",
        dirname=destination_folder,
        advanced=True
    )

    c.file_close_window()  # important


def convert_folder(source_folder, destination_folder):
    """Iterates over files in the specified folder and converts the .prt files to .step files or
    recursively calls the function again if the child i a folder"""

    # delete step folder first
    if RESET_DESTINATION_FOLDER:
        shutil.rmtree(destination_folder, ignore_errors=True, onerror=None)

    for dirpath, _dirnames, filenames in os.walk(source_folder):
        for filename in filenames:
            if filename.split('.')[1].lower() != 'prt':
                continue

            relpath = os.path.relpath(dirpath, source_folder)
            destination_path = os.path.join(destination_folder, relpath)
            if not os.path.exists(destination_path):
                os.makedirs(destination_path, exist_ok=True)
            file = os.path.join(dirpath, filename)  # .replace("\\","/")
            convert_part_to_step(file, destination_path)

if __name__ == "__main__":
    #SRC = r"C:\Users\i00110578\projects\AIAx-Use-Case-1\datasets\MMM\flange-custom-created\flange-custom-extra-samples"
    #DEST = r"C:\Users\i00110578\projects\AIAx-Use-Case-1\datasets\MMM\flange-custom-created\flange-custom-extra-samples-stp"
    SRC = r"C:\Users\i00110578\projects\local_datasets\toy3\extras\test"
    DEST = r"C:\Users\i00110578\projects\local_datasets\toy3\extras\test-stp"
    connect_creo_api()
    convert_folder(SRC, DEST)