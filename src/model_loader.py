import importlib
import sys
from os.path import join, exists, isdir, dirname, abspath

ROOT = abspath('/'.join(dirname(__file__).split('/')[0:-1]))
MODELS_DIR = join(ROOT, 'src', 'models')

def load_model(config):
    model_dir = join(MODELS_DIR, config['model'])

    if not exists(model_dir):
        raise Exception('Model not found')

    if isdir(model_dir):
        sys.path.insert(0, model_dir)
        # model = importlib.import_module(f"{config['model']}.model").model
        model = importlib.import_module(f"model")
    else:
        model = importlib.import_module(f"src.models.{config['model']}")

    return model.Model(config)
