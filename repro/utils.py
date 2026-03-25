from __future__ import annotations

import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
from PIL import Image
import torch
from torch import Tensor

from .losses import reshape_yolo_prediction


DEFAULT_ANCHORS: Tuple[Tuple[Tuple[float, float], ...], ...] = (
    ((10.0, 13.0), (16.0, 30.0), (33.0, 23.0)),
    ((30.0, 61.0), (62.0, 45.0), (59.0, 119.0)),
    ((116.0, 90.0), (156.0, 198.0), (373.0, 326.0)),
)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def confusion_matrix(predictions: Tensor, targets: Tensor, num_classes: int) -> Tensor:
    matrix = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for target, prediction in zip(targets.view(-1), predictions.view(-1)):
        matrix[int(target), int(prediction)] += 1
    return matrix


def classwise_report(matrix: Tensor, class_names: Sequence[str]) -> List[Dict[str, float]]:
    matrix = matrix.to(torch.float32)
    report: List[Dict[str, float]] = []
    for class_index, class_name in enumerate(class_names):
        tp = matrix[class_index, class_index].item()
        fp = matrix[:, class_index].sum().item() - tp
        fn = matrix[class_index, :].sum().item() - tp
        support = matrix[class_index, :].sum().item()
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        report.append(
            {
                "class_name": class_name,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "support": support,
            }
        )
    return report


def summarize_confusion_matrix(matrix: Tensor) -> Dict[str, float]:
    matrix = matrix.to(torch.float32)
    correct = matrix.diag().sum().item()
    total = matrix.sum().item()
    accuracy = correct / total if total else 0.0

    per_class_precision: List[float] = []
    per_class_recall: List[float] = []
    per_class_f1: List[float] = []
    for class_index in range(matrix.shape[0]):
        tp = matrix[class_index, class_index].item()
        fp = matrix[:, class_index].sum().item() - tp
        fn = matrix[class_index, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class_precision.append(precision)
        per_class_recall.append(recall)
        per_class_f1.append(f1)

    return {
        "accuracy": accuracy,
        "precision_macro": sum(per_class_precision) / len(per_class_precision),
        "recall_macro": sum(per_class_recall) / len(per_class_recall),
        "f1_macro": sum(per_class_f1) / len(per_class_f1),
        "support": total,
    }


def save_json(payload: object, output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def save_rows_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return output_path

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def save_confusion_matrix_csv(matrix: Tensor, class_names: Sequence[str], output_path: Path) -> Path:
    rows: List[Dict[str, object]] = []
    matrix_list = matrix.tolist()
    for true_index, class_name in enumerate(class_names):
        row: Dict[str, object] = {"true_class": class_name}
        for pred_index, pred_name in enumerate(class_names):
            row[pred_name] = int(matrix_list[true_index][pred_index])
        rows.append(row)
    return save_rows_csv(rows, output_path)


def aggregate_predictions_by_key(
    rows: Sequence[Dict[str, object]],
    num_classes: int,
    key: str = "video_id",
) -> List[Dict[str, object]]:
    grouped: Dict[str, Dict[str, object]] = {}
    prob_keys = [f"prob_{index}" for index in range(num_classes)]

    for row in rows:
        group_id = str(row.get(key, "unknown"))
        if group_id not in grouped:
            grouped[group_id] = {
                key: group_id,
                "target_id": int(row["target_id"]),
                "target_name": row["target_name"],
                "subject_id": row.get("subject_id", ""),
                "split": row.get("split", ""),
                "count": 0,
                **{prob_key: 0.0 for prob_key in prob_keys},
            }

        item = grouped[group_id]
        item["count"] = int(item["count"]) + 1
        for prob_key in prob_keys:
            item[prob_key] = float(item[prob_key]) + float(row[prob_key])

    aggregated: List[Dict[str, object]] = []
    for group_id, item in grouped.items():
        count = max(int(item["count"]), 1)
        probs = [float(item[f"prob_{index}"]) / count for index in range(num_classes)]
        pred_id = int(np.argmax(np.asarray(probs, dtype=np.float32)))
        output = dict(item)
        for index, value in enumerate(probs):
            output[f"prob_{index}"] = value
        output["pred_id"] = pred_id
        aggregated.append(output)

    aggregated.sort(key=lambda row: str(row[key]))
    return aggregated


def boxes_iou(box: Tensor, boxes: Tensor) -> Tensor:
    x1 = torch.maximum(box[0], boxes[:, 0])
    y1 = torch.maximum(box[1], boxes[:, 1])
    x2 = torch.minimum(box[2], boxes[:, 2])
    y2 = torch.minimum(box[3], boxes[:, 3])

    inter_w = (x2 - x1).clamp(min=0)
    inter_h = (y2 - y1).clamp(min=0)
    inter_area = inter_w * inter_h

    box_area = (box[2] - box[0]).clamp(min=0) * (box[3] - box[1]).clamp(min=0)
    boxes_area = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
    union = box_area + boxes_area - inter_area
    return inter_area / union.clamp_min(1e-6)


def non_max_suppression(boxes: Tensor, scores: Tensor, iou_threshold: float) -> Tensor:
    if boxes.numel() == 0:
        return torch.zeros((0,), dtype=torch.long, device=boxes.device)

    order = scores.argsort(descending=True)
    keep: List[int] = []

    while order.numel() > 0:
        current = int(order[0].item())
        keep.append(current)
        if order.numel() == 1:
            break
        remaining = order[1:]
        ious = boxes_iou(boxes[current], boxes[remaining])
        order = remaining[ious <= iou_threshold]

    return torch.tensor(keep, dtype=torch.long, device=boxes.device)


@torch.no_grad()
def decode_yolo_predictions(
    predictions: Sequence[Tensor],
    anchors: Sequence[Sequence[Tuple[float, float]]],
    num_classes: int,
    image_size: int,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.45,
) -> List[Tensor]:
    batch_size = predictions[0].shape[0]
    anchors_per_scale = len(anchors[0])
    device = predictions[0].device
    outputs: List[Tensor] = []

    for batch_index in range(batch_size):
        batch_boxes: List[Tensor] = []
        for scale_index, pred in enumerate(predictions):
            pred = reshape_yolo_prediction(pred, num_classes, anchors_per_scale)
            pred_scale = pred[batch_index]
            _, height, width, _ = pred_scale.shape

            grid_y, grid_x = torch.meshgrid(
                torch.arange(height, device=device),
                torch.arange(width, device=device),
                indexing="ij",
            )
            grid = torch.stack((grid_x, grid_y), dim=-1).float()
            anchor_tensor = torch.tensor(anchors[scale_index], device=device).view(anchors_per_scale, 1, 1, 2)

            xy = (torch.sigmoid(pred_scale[..., 0:2]) + grid.unsqueeze(0)) / torch.tensor(
                [width, height],
                device=device,
            )
            wh = torch.exp(pred_scale[..., 2:4]) * anchor_tensor / float(image_size)
            obj = torch.sigmoid(pred_scale[..., 4])
            cls_prob = torch.sigmoid(pred_scale[..., 5:])
            score, cls_index = (obj.unsqueeze(-1) * cls_prob).max(dim=-1)

            mask = score > conf_threshold
            if not mask.any():
                continue

            selected_xy = xy[mask]
            selected_wh = wh[mask]
            selected_scores = score[mask]
            selected_cls = cls_index[mask].float()

            x1y1 = selected_xy - selected_wh / 2.0
            x2y2 = selected_xy + selected_wh / 2.0
            boxes = torch.cat([x1y1, x2y2], dim=-1) * float(image_size)
            batch_boxes.append(torch.cat([boxes, selected_scores[:, None], selected_cls[:, None]], dim=-1))

        if not batch_boxes:
            outputs.append(torch.zeros((0, 6), device=device))
            continue

        detections = torch.cat(batch_boxes, dim=0)
        kept: List[Tensor] = []
        for class_index in detections[:, 5].unique():
            class_mask = detections[:, 5] == class_index
            class_dets = detections[class_mask]
            keep = non_max_suppression(class_dets[:, :4], class_dets[:, 4], iou_threshold)
            kept.append(class_dets[keep])
        outputs.append(torch.cat(kept, dim=0) if kept else torch.zeros((0, 6), device=device))

    return outputs


def save_cam_overlay(image: Image.Image, cam: np.ndarray, output_path: Path, alpha: float = 0.35) -> Path:
    image = image.convert("RGB")
    image_np = np.asarray(image, dtype=np.float32)
    cam = np.clip(cam, 0.0, 1.0)
    cam_rgb = np.stack(
        [
            cam * 255.0,
            np.sqrt(cam) * 180.0,
            (1.0 - cam) * 120.0,
        ],
        axis=-1,
    )
    overlay = np.clip((1.0 - alpha) * image_np + alpha * cam_rgb, 0.0, 255.0).astype(np.uint8)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(overlay).save(output_path)
    return output_path
