from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torchvision import models


class ConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: Optional[int] = None,
        negative_slope: float = 0.1,
    ) -> None:
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(negative_slope=negative_slope, inplace=True),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int, hidden_channels: int) -> None:
        super().__init__()
        self.conv1 = ConvBNAct(channels, hidden_channels, kernel_size=1)
        self.conv2 = ConvBNAct(hidden_channels, channels, kernel_size=3)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.conv2(self.conv1(x))


class ResidualStage(nn.Module):
    def __init__(self, channels: int, hidden_channels: int, repeats: int) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[ResidualBlock(channels, hidden_channels) for _ in range(repeats)])

    def forward(self, x: Tensor) -> Tensor:
        return self.blocks(x)


class Darknet53Backbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            ConvBNAct(3, 32, kernel_size=3),
            ConvBNAct(32, 64, kernel_size=3),
        )

        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage1 = ResidualStage(64, 32, repeats=1)

        self.transition1 = ConvBNAct(64, 128, kernel_size=3)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage2 = ResidualStage(128, 64, repeats=2)

        self.transition2 = ConvBNAct(128, 256, kernel_size=3)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage3 = ResidualStage(256, 128, repeats=8)

        self.transition3 = ConvBNAct(256, 512, kernel_size=3)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage4 = ResidualStage(512, 256, repeats=8)

        self.transition4 = ConvBNAct(512, 1024, kernel_size=3)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.stage5 = ResidualStage(1024, 512, repeats=4)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)

        x = self.pool1(x)
        x = self.stage1(x)

        x = self.transition1(x)
        x = self.pool2(x)
        x = self.stage2(x)

        x = self.transition2(x)
        x = self.pool3(x)
        c3 = self.stage3(x)

        x = self.transition3(c3)
        x = self.pool4(x)
        c4 = self.stage4(x)

        x = self.transition4(c4)
        x = self.pool5(x)
        c5 = self.stage5(x)
        return c3, c4, c5


class SpatialAttentionGate(nn.Module):
    def __init__(self, channels: int, gamma_init: float = 1.0) -> None:
        super().__init__()
        self.score = nn.Conv2d(channels, 1, kernel_size=1)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init)))

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        batch_size, _, height, width = x.shape
        logits = self.score(x).reshape(batch_size, 1, height * width)
        weights = torch.softmax(logits, dim=-1).reshape(batch_size, 1, height, width)
        attended = x * weights
        return x + self.gamma * attended, weights


class DetectionHeadBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int) -> None:
        super().__init__()
        expanded_channels = hidden_channels * 2
        self.conv1 = ConvBNAct(in_channels, hidden_channels, kernel_size=1)
        self.conv2 = ConvBNAct(hidden_channels, expanded_channels, kernel_size=3)
        self.conv3 = ConvBNAct(expanded_channels, hidden_channels, kernel_size=1)
        self.conv4 = ConvBNAct(hidden_channels, expanded_channels, kernel_size=3)
        self.conv5 = ConvBNAct(expanded_channels, hidden_channels, kernel_size=1)
        self.pred_prep = ConvBNAct(hidden_channels, expanded_channels, kernel_size=3)
        self.pred = nn.Conv2d(expanded_channels, out_channels, kernel_size=1)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        route = self.conv5(x)
        pred = self.pred(self.pred_prep(route))
        return route, pred


class AttentionYOLOv3Drowsiness(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        detection_num_classes: Optional[int] = None,
        anchors_per_scale: int = 3,
        enable_detection: bool = True,
        classifier_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.detection_num_classes = detection_num_classes if detection_num_classes is not None else num_classes
        self.anchors_per_scale = anchors_per_scale
        self.enable_detection = enable_detection

        self.backbone = Darknet53Backbone()
        self.attention = SpatialAttentionGate(1024)

        self.classifier_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(1024, num_classes)

        if enable_detection:
            det_out = anchors_per_scale * (5 + self.detection_num_classes)
            self.head_large = DetectionHeadBlock(1024, 512, det_out)
            self.up_large = nn.Sequential(
                ConvBNAct(512, 256, kernel_size=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )
            self.head_medium = DetectionHeadBlock(256 + 512, 256, det_out)
            self.up_medium = nn.Sequential(
                ConvBNAct(256, 128, kernel_size=1),
                nn.Upsample(scale_factor=2, mode="nearest"),
            )
            self.head_small = DetectionHeadBlock(128 + 256, 128, det_out)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        c3, c4, c5 = self.backbone(x)
        attended_c5, attention_map = self.attention(c5)

        pooled = self.classifier_pool(attended_c5).flatten(1)
        pooled = self.classifier_dropout(pooled)
        logits = self.classifier(pooled)

        outputs: Dict[str, Tensor] = {
            "logits": logits,
            "features": attended_c5,
            "attention_map": F.interpolate(
                attention_map,
                size=x.shape[-2:],
                mode="bilinear",
                align_corners=False,
            ),
        }

        if self.enable_detection:
            route_large, pred_large = self.head_large(attended_c5)
            up_large = self.up_large(route_large)

            route_medium, pred_medium = self.head_medium(torch.cat([up_large, c4], dim=1))
            up_medium = self.up_medium(route_medium)

            _, pred_small = self.head_small(torch.cat([up_medium, c3], dim=1))
            outputs["det_preds"] = [pred_small, pred_medium, pred_large]

        return outputs

    @torch.no_grad()
    def compute_cam(
        self,
        features: Tensor,
        class_indices: Optional[Tensor] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if class_indices is None:
            pooled = self.classifier_pool(features).flatten(1)
            logits = self.classifier(pooled)
            class_indices = logits.argmax(dim=1)

        weights = self.classifier.weight[class_indices]
        cam = (features * weights[:, :, None, None]).sum(dim=1)
        cam = F.relu(cam)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)

        if input_size is not None:
            cam = F.interpolate(cam.unsqueeze(1), size=input_size, mode="bilinear", align_corners=False).squeeze(1)
        return cam

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"detection_num_classes={self.detection_num_classes}, "
            f"anchors_per_scale={self.anchors_per_scale}, "
            f"enable_detection={self.enable_detection}"
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        for parameter in self.backbone.parameters():
            parameter.requires_grad = trainable


class AttentionResNetClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 2,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        classifier_dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.pretrained = pretrained

        if backbone_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            backbone = models.resnet18(weights=weights)
            feature_channels = 512
        elif backbone_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            backbone = models.resnet50(weights=weights)
            feature_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone_name: {backbone_name}")

        self.stem = nn.Sequential(
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
        )
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.attention = SpatialAttentionGate(feature_channels)
        self.classifier_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier_dropout = nn.Dropout(classifier_dropout)
        self.classifier = nn.Linear(feature_channels, num_classes)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        input_size = x.shape[-2:]
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        features = self.layer4(x)

        attended_features, attention_map = self.attention(features)
        pooled = self.classifier_pool(attended_features).flatten(1)
        pooled = self.classifier_dropout(pooled)
        logits = self.classifier(pooled)
        return {
            "logits": logits,
            "features": attended_features,
            "attention_map": F.interpolate(
                attention_map,
                size=input_size,
                mode="bilinear",
                align_corners=False,
            ),
        }

    @torch.no_grad()
    def compute_cam(
        self,
        features: Tensor,
        class_indices: Optional[Tensor] = None,
        input_size: Optional[Tuple[int, int]] = None,
    ) -> Tensor:
        if class_indices is None:
            pooled = self.classifier_pool(features).flatten(1)
            logits = self.classifier(pooled)
            class_indices = logits.argmax(dim=1)

        weights = self.classifier.weight[class_indices]
        cam = (features * weights[:, :, None, None]).sum(dim=1)
        cam = F.relu(cam)
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / cam.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        if input_size is not None:
            cam = F.interpolate(cam.unsqueeze(1), size=input_size, mode="bilinear", align_corners=False).squeeze(1)
        return cam

    def extra_repr(self) -> str:
        return (
            f"num_classes={self.num_classes}, "
            f"backbone_name={self.backbone_name}, "
            f"pretrained={self.pretrained}"
        )

    def set_backbone_trainable(self, trainable: bool) -> None:
        for module in (self.stem, self.layer1, self.layer2, self.layer3, self.layer4):
            for parameter in module.parameters():
                parameter.requires_grad = trainable
