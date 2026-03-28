from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


def reshape_yolo_prediction(prediction: Tensor, num_classes: int, anchors_per_scale: int) -> Tensor:
    batch_size, _, height, width = prediction.shape
    prediction = prediction.view(batch_size, anchors_per_scale, 5 + num_classes, height, width)
    prediction = prediction.permute(0, 1, 3, 4, 2).contiguous()
    return prediction


def wh_iou(gt_wh: Tensor, anchors_wh: Tensor) -> Tensor:
    gt_wh = gt_wh[:, None, :]
    anchors_wh = anchors_wh[None, :, :]
    inter = torch.minimum(gt_wh, anchors_wh).prod(dim=-1)
    union = gt_wh.prod(dim=-1) + anchors_wh.prod(dim=-1) - inter
    return inter / union.clamp_min(1e-6)


class YoloDetectionLoss(nn.Module):
    def __init__(
        self,
        anchors: Sequence[Sequence[Tuple[float, float]]],
        num_classes: int,
        image_size: int = 416,
        box_weight: float = 1.0,
        obj_weight: float = 1.0,
        noobj_weight: float = 0.25,
        cls_weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.anchors = tuple(tuple(anchor for anchor in scale) for scale in anchors)
        self.num_classes = num_classes
        self.image_size = image_size
        self.box_weight = box_weight
        self.obj_weight = obj_weight
        self.noobj_weight = noobj_weight
        self.cls_weight = cls_weight
        self.anchors_per_scale = len(self.anchors[0])

    def forward(self, predictions: Sequence[Tensor], targets: List[Tensor]) -> Dict[str, Tensor]:
        device = predictions[0].device
        dtype = predictions[0].dtype
        preds = [reshape_yolo_prediction(pred, self.num_classes, self.anchors_per_scale) for pred in predictions]

        obj_targets: List[Tensor] = []
        box_xy_targets: List[Tensor] = []
        box_wh_targets: List[Tensor] = []
        cls_targets: List[Tensor] = []
        pos_masks: List[Tensor] = []

        for pred in preds:
            obj_targets.append(torch.zeros(pred.shape[:-1], device=device, dtype=dtype))
            box_xy_targets.append(torch.zeros((*pred.shape[:-1], 2), device=device, dtype=dtype))
            box_wh_targets.append(torch.zeros((*pred.shape[:-1], 2), device=device, dtype=dtype))
            cls_targets.append(torch.zeros((*pred.shape[:-1], self.num_classes), device=device, dtype=dtype))
            pos_masks.append(torch.zeros(pred.shape[:-1], device=device, dtype=torch.bool))

        flat_anchors = torch.tensor(
            [anchor for scale in self.anchors for anchor in scale],
            dtype=dtype,
            device=device,
        )

        for batch_index, target in enumerate(targets):
            if target.numel() == 0:
                continue

            target = target.to(device=device, dtype=dtype)
            gt_wh_pixels = target[:, 3:5] * float(self.image_size)
            best_anchor_indices = wh_iou(gt_wh_pixels, flat_anchors).argmax(dim=1)

            for gt_row, flat_anchor_index in zip(target, best_anchor_indices):
                scale_index = int(flat_anchor_index.item() // self.anchors_per_scale)
                anchor_index = int(flat_anchor_index.item() % self.anchors_per_scale)

                pred = preds[scale_index]
                _, _, height, width, _ = pred.shape
                gx = gt_row[1] * width
                gy = gt_row[2] * height
                gw = gt_row[3] * width
                gh = gt_row[4] * height

                grid_x = min(width - 1, max(0, int(gx.item())))
                grid_y = min(height - 1, max(0, int(gy.item())))

                obj_targets[scale_index][batch_index, anchor_index, grid_y, grid_x] = 1.0
                pos_masks[scale_index][batch_index, anchor_index, grid_y, grid_x] = True
                box_xy_targets[scale_index][batch_index, anchor_index, grid_y, grid_x, 0] = gx - grid_x
                box_xy_targets[scale_index][batch_index, anchor_index, grid_y, grid_x, 1] = gy - grid_y

                box_wh_targets[scale_index][batch_index, anchor_index, grid_y, grid_x, 0] = gw
                box_wh_targets[scale_index][batch_index, anchor_index, grid_y, grid_x, 1] = gh

                class_index = int(gt_row[0].item())
                if 0 <= class_index < self.num_classes:
                    cls_targets[scale_index][batch_index, anchor_index, grid_y, grid_x, class_index] = 1.0

        loss_box = torch.zeros((), device=device, dtype=dtype)
        loss_obj = torch.zeros((), device=device, dtype=dtype)
        loss_cls = torch.zeros((), device=device, dtype=dtype)

        for scale_index, pred in enumerate(preds):
            pos_mask = pos_masks[scale_index]
            obj_target = obj_targets[scale_index]

            pred_xy = pred[..., 0:2]
            pred_wh = pred[..., 2:4]
            pred_obj = pred[..., 4]
            pred_cls = pred[..., 5:]

            _, anchors_per_scale, height, width, _ = pred.shape
            anchor_tensor = torch.tensor(self.anchors[scale_index], dtype=dtype, device=device).view(
                1, anchors_per_scale, 1, 1, 2
            )
            anchor_tensor = anchor_tensor.clone()
            anchor_tensor[..., 0] = (anchor_tensor[..., 0] / self.image_size) * width
            anchor_tensor[..., 1] = (anchor_tensor[..., 1] / self.image_size) * height

            if pos_mask.any():
                positive_count = pos_mask.sum().to(dtype=dtype).clamp_min(1.0)
                pred_xy_decoded = pred_xy.sigmoid()
                expanded_anchor_tensor = anchor_tensor.expand(pred.shape[0], -1, height, width, -1)
                target_wh_log = torch.log(
                    box_wh_targets[scale_index].clamp_min(1e-6) / expanded_anchor_tensor.clamp_min(1e-6)
                )
                target_wh_log = target_wh_log.clamp(min=-6.0, max=6.0)

                loss_box = loss_box + (
                    F.mse_loss(
                        pred_xy_decoded[pos_mask],
                        box_xy_targets[scale_index][pos_mask],
                        reduction="sum",
                    )
                    + F.mse_loss(
                        pred_wh[pos_mask],
                        target_wh_log[pos_mask],
                        reduction="sum",
                    )
                ) / positive_count

                if self.cls_weight > 0.0 and pred_cls.shape[-1] > 0:
                    cls_target_ids = cls_targets[scale_index][pos_mask].argmax(dim=-1)
                    loss_cls = loss_cls + F.cross_entropy(
                        pred_cls[pos_mask],
                        cls_target_ids,
                        reduction="mean",
                    )

            negative_mask = ~pos_mask
            if pos_mask.any():
                positive_obj_loss = F.binary_cross_entropy_with_logits(
                    pred_obj[pos_mask],
                    obj_target[pos_mask],
                    reduction="mean",
                )
            else:
                positive_obj_loss = torch.zeros((), device=device, dtype=dtype)

            if negative_mask.any():
                negative_obj_loss = F.binary_cross_entropy_with_logits(
                    pred_obj[negative_mask],
                    obj_target[negative_mask],
                    reduction="mean",
                )
            else:
                negative_obj_loss = torch.zeros((), device=device, dtype=dtype)

            loss_obj = loss_obj + self.obj_weight * positive_obj_loss + self.noobj_weight * negative_obj_loss

        loss_box = loss_box * self.box_weight
        loss_cls = loss_cls * self.cls_weight
        total = loss_box + loss_obj + loss_cls

        return {
            "loss": total,
            "loss_box": loss_box.detach(),
            "loss_obj": loss_obj.detach(),
            "loss_cls": loss_cls.detach(),
        }
