from pathlib import Path
import json
import random

import numpy as np
import torch
import yaml
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[1]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def project_path(value):
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def load_config(path, task):
    with project_path(path).open(encoding="utf-8") as stream:
        cfg = yaml.safe_load(stream)
    if not isinstance(cfg, dict) or cfg.get("task") != task:
        raise ValueError(f"Expected a {task!r} configuration")
    return cfg


def get_device(name):
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(name)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def read_rgb(path):
    with Image.open(path) as image:
        # All annotation coordinates must refer to this EXIF-corrected view.
        return ImageOps.exif_transpose(image).convert("RGB")


def image_tensor(image, size=None):
    if size is not None:
        height, width = map(int, size)
        image = image.resize((width, height), Image.Resampling.BILINEAR)
    array = np.array(image, dtype=np.float32, copy=True) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def tensor_image(tensor):
    array = tensor.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy()
    return Image.fromarray((array * 255).round().astype(np.uint8))


def list_images(folder):
    folder = project_path(folder)
    if not folder.is_dir():
        raise FileNotFoundError(f"Image folder does not exist: {folder}")
    images = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
    if not images:
        raise ValueError(f"No supported images in {folder}")
    stems = [p.stem for p in images]
    if len(stems) != len(set(stems)):
        raise ValueError(f"Duplicate image stems in {folder}")
    return images


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
