from pathlib import Path

import cv2
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


print("Python environment test")
print("-" * 50)

print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print(
        "VRAM:",
        round(
            torch.cuda.get_device_properties(0).total_memory
            / 1024 ** 3,
            2
        ),
        "GB"
    )

print("OpenCV:", cv2.__version__)

with open(
    CONFIG_PATH,
    "r",
    encoding="utf-8"
) as file:
    config = yaml.safe_load(file)

data_root = Path(config["paths"]["data_root"])

print("Data root:", data_root)
print("Data root exists:", data_root.exists())