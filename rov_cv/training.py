import argparse
import math

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .common import get_device, load_config, project_path, seed_everything, write_json
from .datasets import detection_collate, make_dataset
from .models import build_model


def pixel_loss(prediction, target, task):
    if task == "enhancement":
        return F.l1_loss(prediction, target)
    bce = F.binary_cross_entropy_with_logits(prediction, target)
    probability = prediction.sigmoid()
    dims = (1, 2, 3)
    intersection = (probability * target).sum(dims)
    dice = (2 * intersection + 1) / (probability.sum(dims) + target.sum(dims) + 1)
    return bce + (1 - dice).mean()


@torch.inference_mode()
def validate(model, loader, cfg, device):
    model.eval()
    task = cfg["task"]
    tp = fp = fn = 0
    intersection = predicted_pixels = target_pixels = absolute_error = elements = 0.0
    for images, targets in loader:
        if task == "detection":
            from torchvision.ops import box_iou
            predictions = model([image.to(device) for image in images])
            threshold = float(cfg.get("inference", {}).get("score_threshold", 0.5))
            for prediction, target in zip(predictions, targets):
                keep = prediction["scores"].cpu() >= threshold
                boxes = prediction["boxes"].cpu()[keep]
                labels = prediction["labels"].cpu()[keep]
                scores = prediction["scores"].cpu()[keep]
                overlaps = box_iou(boxes, target["boxes"])
                matched = set()
                for index in scores.argsort(descending=True).tolist():
                    candidates = [j for j in range(len(target["boxes"])) if j not in matched and labels[index] == target["labels"][j]]
                    best = max(candidates, key=lambda j: float(overlaps[index, j])) if candidates else None
                    if best is not None and overlaps[index, best] >= 0.5:
                        matched.add(best)
                        tp += 1
                    else:
                        fp += 1
                fn += len(target["boxes"]) - len(matched)
        else:
            prediction = model(images.to(device))
            targets = targets.to(device)
            if task == "segmentation":
                threshold = float(cfg.get("inference", {}).get("mask_threshold", 0.5))
                mask = prediction.sigmoid() >= threshold
                truth = targets > 0.5
                intersection += (mask & truth).sum().item()
                predicted_pixels += mask.sum().item()
                target_pixels += truth.sum().item()
            else:
                absolute_error += (prediction - targets).abs().sum().item()
                elements += targets.numel()
    if task == "detection":
        precision, recall = tp / max(tp + fp, 1), tp / max(tp + fn, 1)
        f1 = 2 * tp / max(2 * tp + fp + fn, 1)
        return f1, {"precision_at_iou_0.5": precision, "recall_at_iou_0.5": recall, "f1_at_iou_0.5": f1, "tp": tp, "fp": fp, "fn": fn}
    if task == "segmentation":
        denominator = predicted_pixels + target_pixels
        dice = 2 * intersection / denominator if denominator else 1.0
        union = denominator - intersection
        iou = intersection / union if union else 1.0
        return dice, {"pixel_dice": dice, "pixel_iou": iou}
    mae = absolute_error / max(elements, 1)
    return -mae, {"mae_rgb_0_1": mae}


def train(task):
    parser = argparse.ArgumentParser(description=f"Train only the {task} model")
    parser.add_argument("--config", default=f"configs/{task}.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config, task)
    seed_everything(int(cfg.get("seed", 42)))
    device = get_device(cfg.get("device", "auto"))
    settings = cfg["train"]
    if int(settings["epochs"]) < 1:
        raise ValueError("epochs must be at least 1")
    model = build_model(cfg).to(device)
    loaders = {}
    for split in ("train", "val"):
        loaders[split] = DataLoader(
            make_dataset(cfg, split), batch_size=int(settings["batch_size"]),
            shuffle=split == "train", num_workers=int(settings.get("workers", 0)),
            collate_fn=detection_collate if task == "detection" else None,
        )
    output = project_path(settings["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if any((output / name).exists() for name in ("last.pt", "best.pt", "history.json")):
        raise FileExistsError(f"Training artifacts already exist in {output}; choose a new output_dir")
    write_json(output / "config.json", cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(settings["lr"]), weight_decay=float(settings["weight_decay"]))
    best_score, history = -math.inf, []
    for epoch in range(1, int(settings["epochs"]) + 1):
        model.train()
        total_loss, batches = 0.0, 0
        for images, targets in tqdm(loaders["train"], desc=f"{task} {epoch}/{settings['epochs']}"):
            optimizer.zero_grad(set_to_none=True)
            if task == "detection":
                images = [image.to(device) for image in images]
                targets = [{key: value.to(device) for key, value in target.items()} for target in targets]
                loss = sum(model(images, targets).values())
            else:
                loss = pixel_loss(model(images.to(device)), targets.to(device), task)
            if not torch.isfinite(loss).item():
                raise FloatingPointError("Non-finite training loss; check data and learning rate")
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batches += 1
        score, metrics = validate(model, loaders["val"], cfg, device)
        record = {"epoch": epoch, "train_loss": total_loss / max(batches, 1), **metrics}
        history.append(record)
        checkpoint = {
            "format_version": 1, "task": task, "epoch": epoch, "config": cfg,
            "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
            "metrics": record,
        }
        torch.save(checkpoint, output / "last.pt")
        if score > best_score:
            best_score = score
            torch.save(checkpoint, output / "best.pt")
        write_json(output / "history.json", history)
        print(record)


# Windows DataLoader entry points live in the separate guarded train_* modules.
