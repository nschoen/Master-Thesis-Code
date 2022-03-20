import os
import subprocess

MANIFOLD_SIMPLIFICATION_SCRIPT = "/home/i00110578/projects/ManifoldPlus/build/manifold"

def watetight_mesh(src, dest):
    if os.path.isfile(dest):
        return
    subprocess.call([MANIFOLD_SIMPLIFICATION_SCRIPT, '--input', src, '--output', dest, '--depth', '8'])