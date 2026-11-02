from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F

from .common import get_device, image_tensor, project_path, read_rgb, tensor_image, write_json
from .models import build_model


class VisionPipeline:
    """Loads only explicitly supplied weights; enhancement is opt-in."""

    def __init__(self, detection=None, segmentation=None, enhancement=None, device="auto"):
        self.device = get_device(device)
        self.models = {}
        for task, path in (("detection", detection), ("segmentation", segmentation), ("enhancement", enhancement)):
            if path is None:
                continue
            checkpoint = torch.load(project_path(path), map_location="cpu", weights_only=True)
            if checkpoint.get("format_version") != 1 or checkpoint.get("task") != task:
                raise ValueError(f"Checkpoint does not match task {task}")
            cfg = checkpoint["config"]
            if cfg.get("task") != task:
                raise ValueError("Checkpoint configuration task mismatch")
            model = build_model(cfg)
            model.load_state_dict(checkpoint["model_state"], strict=True)
            model.to(self.device).eval()
            self.models[task] = (model, cfg)
        if not self.models:
            raise ValueError("Provide at least one checkpoint")

    @torch.inference_mode()
    def predict(self, image_path, output_dir, score_threshold=None, mask_threshold=None):
        """Process a single image; write result.json and optional raster artifacts."""
        for name, value in (("score_threshold", score_threshold), ("mask_threshold", mask_threshold)):
            if value is not None and not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        source = project_path(image_path)
        original = read_rgb(source)
        width, height = original.size
        output = project_path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        artifact_names = ("result.json", "enhanced.png", "crack_mask.png", "overlay.png")
        if any((output / name).exists() for name in artifact_names):
            raise FileExistsError(f"Result files already exist in {output}; choose a new output_dir")
        working = original
        result = {
            "schema_version": "1.0", "image_path": str(source.resolve()),
            "width": width, "height": height, "coordinate_space": "exif_corrected_original_pixels",
            "bbox_format": "xyxy", "enhancement_applied": "enhancement" in self.models,
            "detection_executed": "detection" in self.models,
            "segmentation_executed": "segmentation" in self.models,
            "detections": [], "segmentation": None, "artifacts": {},
        }
        if "enhancement" in self.models:
            model, cfg = self.models["enhancement"]
            tensor = image_tensor(original, cfg["model"]["image_size"]).unsqueeze(0).to(self.device)
            enhanced = F.interpolate(model(tensor), size=(height, width), mode="bilinear", align_corners=False)
            working = tensor_image(enhanced[0])
            working.save(output / "enhanced.png")
            result["artifacts"]["enhanced_image"] = "enhanced.png"
        if "detection" in self.models:
            model, cfg = self.models["detection"]
            threshold = float(score_threshold if score_threshold is not None else cfg.get("inference", {}).get("score_threshold", 0.5))
            prediction = model([image_tensor(working).to(self.device)])[0]
            result["detection_score_threshold"] = threshold
            for box, label, score in zip(prediction["boxes"].cpu().tolist(), prediction["labels"].cpu().tolist(), prediction["scores"].cpu().tolist()):
                if score >= threshold:
                    result["detections"].append({"class_id": label, "class_name": cfg["model"]["classes"][label], "score": score, "bbox_xyxy": box})
        overlay = working.copy()
        if "segmentation" in self.models:
            model, cfg = self.models["segmentation"]
            threshold = float(mask_threshold if mask_threshold is not None else cfg.get("inference", {}).get("mask_threshold", 0.5))
            tensor = image_tensor(working, cfg["model"]["image_size"]).unsqueeze(0).to(self.device)
            probability = F.interpolate(model(tensor).sigmoid(), size=(height, width), mode="bilinear", align_corners=False)
            mask = probability[0, 0].cpu().numpy() >= threshold
            Image.fromarray(mask.astype(np.uint8) * 255).save(output / "crack_mask.png")
            pixels = np.array(overlay, copy=True)
            pixels[mask] = (0.55 * pixels[mask] + 0.45 * np.array([255, 0, 0])).astype(np.uint8)
            overlay = Image.fromarray(pixels)
            result["segmentation"] = {
                "class_name": "crack", "mask_path": "crack_mask.png", "threshold": threshold,
                "crack_pixels": int(mask.sum()), "crack_area_ratio": float(mask.mean()),
            }
            result["artifacts"]["crack_mask"] = "crack_mask.png"
        draw = ImageDraw.Draw(overlay)
        for item in result["detections"]:
            box = item["bbox_xyxy"]
            draw.rectangle(box, outline=(0, 255, 0), width=3)
            # Use numeric IDs in the raster label to avoid system font dependencies.
            draw.text((box[0], box[1]), f"{item['class_id']} {item['score']:.2f}", fill=(255, 255, 0))
        overlay.save(output / "overlay.png")
        result["artifacts"]["overlay"] = "overlay.png"
        write_json(output / "result.json", result)
        return result
