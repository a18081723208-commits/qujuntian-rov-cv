import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .common import image_tensor, list_images, project_path, read_rgb


class DetectionDataset(Dataset):
    """COCO boxes -> torchvision targets. Categories are mapped by name."""

    def __init__(self, image_dir, annotations, classes):
        self.image_dir = project_path(image_dir)
        with project_path(annotations).open(encoding="utf-8") as stream:
            coco = json.load(stream)
        self.images = coco["images"]
        if not self.images:
            raise ValueError("COCO images list is empty")
        self.mapping = {}
        for category in coco["categories"]:
            if category["name"] not in classes[1:]:
                raise ValueError(f"Unknown foreground category: {category['name']}")
            self.mapping[category["id"]] = classes.index(category["name"])
        self.annotations = {item["id"]: [] for item in self.images}
        if len(self.annotations) != len(self.images):
            raise ValueError("Duplicate COCO image IDs")
        for annotation in coco["annotations"]:
            if annotation.get("iscrowd", 0):
                raise ValueError("iscrowd=1 is not supported by this baseline")
            if annotation["category_id"] not in self.mapping:
                raise ValueError("Annotation refers to an undefined category")
            self.annotations[annotation["image_id"]].append(annotation)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        info = self.images[index]
        relative = Path(info["file_name"])
        path = (self.image_dir / relative).resolve()
        if relative.is_absolute() or not path.is_relative_to(self.image_dir.resolve()):
            raise ValueError("COCO file_name must stay inside image_dir")
        image = read_rgb(path)
        width, height = image.size
        if (int(info["width"]), int(info["height"])) != image.size:
            raise ValueError(f"COCO dimensions do not match EXIF-corrected image: {path}")
        boxes, labels = [], []
        for annotation in self.annotations[info["id"]]:
            x, y, w, h = map(float, annotation["bbox"])
            if not np.isfinite([x, y, w, h]).all() or w <= 0 or h <= 0 or x < 0 or y < 0 or x + w > width or y + h > height:
                raise ValueError(f"Invalid COCO box in image {info['id']}")
            boxes.append([x, y, x + w, y + h])
            labels.append(self.mapping[annotation["category_id"]])
        boxes = torch.tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        target = {
            "boxes": boxes,
            "labels": torch.tensor(labels, dtype=torch.int64),
            "image_id": torch.tensor([info["id"]], dtype=torch.int64),
            "area": (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return image_tensor(image), target


class PairedDataset(Dataset):
    """Flat folders; image stem pairs with a PNG mask or same-stem RGB target."""

    def __init__(self, images, targets, size, task):
        self.images = list_images(images)
        self.size, self.task = size, task
        target_dir = project_path(targets)
        if task == "segmentation":
            self.targets = {image.stem: target_dir / f"{image.stem}.png" for image in self.images}
        else:
            self.targets = {image.stem: image for image in list_images(target_dir)}
        for image in self.images:
            if image.stem not in self.targets or not self.targets[image.stem].is_file():
                raise FileNotFoundError(f"Missing pair for {image.name}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        path = self.images[index]
        image = read_rgb(path)
        target_path = self.targets[path.stem]
        if self.task == "segmentation":
            with Image.open(target_path) as source:
                source = ImageOps.exif_transpose(source)
                if source.mode not in {"1", "L", "P"}:
                    raise ValueError(f"Mask must be a single-channel or indexed binary PNG: {target_path}")
                mask = np.array(source, copy=True)
                if not set(np.unique(mask)).issubset({0, 1, 255}):
                    raise ValueError(f"Mask contains values outside 0/1/255: {target_path}")
                if source.size != image.size:
                    raise ValueError(f"Image/mask dimensions differ: {path}")
                mask_image = Image.fromarray((mask > 0).astype(np.uint8) * 255)
            height, width = self.size
            mask_image = mask_image.resize((width, height), Image.Resampling.NEAREST)
            target = torch.from_numpy((np.array(mask_image) > 0).astype(np.float32)).unsqueeze(0)
        else:
            target_image = read_rgb(target_path)
            if target_image.size != image.size:
                raise ValueError(f"Enhancement pair dimensions differ: {path}")
            target = image_tensor(target_image, self.size)
        return image_tensor(image, self.size), target


def detection_collate(batch):
    return tuple(zip(*batch))


def make_dataset(cfg, split):
    data, task = cfg["data"], cfg["task"]
    if task == "detection":
        return DetectionDataset(data[f"{split}_images"], data[f"{split}_annotations"], cfg["model"]["classes"])
    key = "masks" if task == "segmentation" else "targets"
    return PairedDataset(data[f"{split}_images"], data[f"{split}_{key}"], cfg["model"]["image_size"], task)
