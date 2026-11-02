import torch
from torch import nn
from torch.nn import functional as F


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.ReLU(inplace=True),
    )


class SmallUNet(nn.Module):
    """Binary crack segmentation. Returns unnormalized logits [B, 1, H, W]."""

    def __init__(self, channels=32):
        super().__init__()
        self.enc1 = double_conv(3, channels)
        self.enc2 = double_conv(channels, channels * 2)
        self.bridge = double_conv(channels * 2, channels * 4)
        self.dec2 = double_conv(channels * 6, channels * 2)
        self.dec1 = double_conv(channels * 3, channels)
        self.head = nn.Conv2d(channels, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(F.max_pool2d(e1, 2))
        z = self.bridge(F.max_pool2d(e2, 2))
        z = self.dec2(torch.cat([F.interpolate(z, size=e2.shape[-2:], mode="bilinear", align_corners=False), e2], dim=1))
        z = self.dec1(torch.cat([F.interpolate(z, size=e1.shape[-2:], mode="bilinear", align_corners=False), e1], dim=1))
        return self.head(z)


class ResidualEnhancer(nn.Module):
    """Small paired-data baseline; has no underwater physics or pretrained weights."""

    def __init__(self, channels=32):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(channels, 3, 3, padding=1),
        )

    def forward(self, x):
        return (x + self.body(x)).clamp(0, 1)


def build_model(cfg):
    task, model = cfg["task"], cfg["model"]
    if task == "detection":
        if model["name"] != "fasterrcnn_resnet50_fpn":
            raise ValueError("Unsupported detection model")
        classes = model["classes"]
        if len(classes) < 2 or classes[0] != "background" or len(set(classes)) != len(classes):
            raise ValueError("classes must contain unique names starting with background")
        from torchvision.models.detection import fasterrcnn_resnet50_fpn
        # Explicitly disable both detector and backbone weight downloads.
        return fasterrcnn_resnet50_fpn(
            weights=None, weights_backbone=None, num_classes=len(classes),
            min_size=int(model.get("min_size", 640)), max_size=int(model.get("max_size", 1024)),
        )
    size = model["image_size"]
    if len(size) != 2 or any(int(x) < 4 or int(x) % 4 for x in size):
        raise ValueError("image_size must be [height, width], positive multiples of 4")
    if task == "segmentation" and model["name"] == "unet_small":
        return SmallUNet(int(model.get("base_channels", 32)))
    if task == "enhancement" and model["name"] == "residual_enhancer":
        return ResidualEnhancer(int(model.get("channels", 32)))
    raise ValueError(f"Unsupported model for task {task}")
