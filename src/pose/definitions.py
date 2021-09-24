import os
from pathlib import Path

import torch
import yaml


def load_settings():
    with open(SETTINGS_PATH, "r") as settings_file:
        settings = yaml.load(settings_file, Loader=yaml.FullLoader)
    return settings


_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_PATH = Path(_ROOT_DIR) / "../.."
SETTINGS_PATH = ROOT_PATH / "src/settings.yaml"
SETTINGS = load_settings()
TORCH_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
