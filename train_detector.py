from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor, nn
from torch.utils.data import DataLoader
from torchvision import transforms

from repro.data import ResizeWithPad, YoloDetectionDataset, detection_collate_fn
from repro.losses import YoloDetectionLoss
from repro.model import AttentionYOLOv3Drowsiness
from repro.utils import (
    DEFAULT_ANCHORS,
    aggregate_predictions_by_key,
    classwise_report,
    confusion_matrix,
    save_cam_overlay,
    save_confusion_matrix_csv,
    save_json,
    save_rows_csv,
    set_seed,
    summarize_confusion_matrix,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the full attention YOLOv3 detector.")
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--val-images", type=Path, default=None)
    parser.add_argument("--val-labels", type=Path, default=None)
    parser.add_argument("--test-images", type=Path, default=None)
    parser.add_argument("--test-labels", type=Path, default=None)
    parser.add_argument("--class-names", nargs="+", default=None)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Load official Darknet-53 conv.74 backbone weights for paper_attention_yolo.",
    )
    parser.add_argument(
        "--pretrained-backbone-path",
        type=Path,
        default=None,
        help="Optional local path to Darknet-53 conv.74 weights. If omitted, the weights are downloaded automatically.",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--resize-mode", choices=("stretch", "letterbox"), default="stretch")
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="sgd")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--nesterov", action="store_true", help="Enable Nesterov momentum for SGD.")
    parser.add_argument("--scheduler", choices=("none", "cosine", "multistep"), default="multistep")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument("--lr-milestones", nargs="+", type=int, default=[60, 110])
    parser.add_argument("--lr-gamma", type=float, default=0.1)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--selection-metric", choices=("accuracy", "f1_macro"), default="f1_macro")
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--early-stopping-patience", type=int, default=0)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--limit-batches", type=int, default=0)
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--limit-test-batches", type=int, default=0)
    parser.add_argument("--paper-preset", action="store_true", help="Apply the paper-style detector hyperparameters.")
    parser.add_argument("--cam-samples", type=int, default=8)
    return parser.parse_args()


def apply_paper_detector_preset(args: argparse.Namespace) -> argparse.Namespace:
    if not args.paper_preset:
        return args

    args.image_size = 416
    args.resize_mode = "letterbox"
    args.optimizer = "sgd"
    args.lr = 1e-2
    args.weight_decay = 5e-4
    args.momentum = 0.9
    args.scheduler = "multistep"
    args.lr_milestones = [60, 110]
    args.lr_gamma = 0.1
    args.epochs = 150
    args.batch_size = 40
    args.label_smoothing = 0.0
    args.grad_clip_norm = 10.0
    return args


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


def resolve_limit(limit_specific: int, limit_shared: int) -> int:
    return limit_specific if limit_specific > 0 else limit_shared


def build_loader(
    images_dir: Optional[Path],
    labels_dir: Optional[Path],
    image_size: int,
    batch_size: int,
    resize_mode: str,
    num_workers: int,
    shuffle: bool,
) -> Optional[DataLoader]:
    if images_dir is None or labels_dir is None:
        return None
    dataset = YoloDetectionDataset(
        images_dir=images_dir,
        labels_dir=labels_dir,
        image_size=image_size,
        resize_mode=resize_mode,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=detection_collate_fn,
        pin_memory=False,
    )


def infer_class_names(images_dir: Path, num_classes: int, class_names_arg: Optional[Sequence[str]]) -> List[str]:
    if class_names_arg:
        class_names = list(class_names_arg)
    else:
        class_names = sorted(path.name for path in Path(images_dir).iterdir() if path.is_dir())
        if not class_names:
            class_names = [f"class_{index}" for index in range(num_classes)]
        elif len(class_names) < num_classes:
            class_names = [*class_names, *[f"class_{index}" for index in range(len(class_names), num_classes)]]
    if len(class_names) != num_classes:
        raise ValueError(f"class_names length {len(class_names)} does not match num_classes {num_classes}")
    return class_names


def image_level_labels(targets: List[Tensor], device: torch.device) -> torch.Tensor:
    labels = []
    for target in targets:
        if target.numel() == 0:
            labels.append(0)
        else:
            labels.append(int(target[0, 0].item()))
    return torch.tensor(labels, dtype=torch.long, device=device)


def create_optimizer(args: argparse.Namespace, model: torch.nn.Module) -> torch.optim.Optimizer:
    parameters = list(model.parameters())
    if args.optimizer == "adam":
        return torch.optim.Adam(parameters, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(parameters, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def create_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    if args.scheduler == "none":
        return None
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(args.epochs, 1),
            eta_min=args.min_lr,
        )
    if args.scheduler == "multistep":
        milestones = sorted(set(int(milestone) for milestone in args.lr_milestones if int(milestone) > 0))
        return torch.optim.lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=args.lr_gamma,
        )
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def create_detection_loss(args: argparse.Namespace) -> YoloDetectionLoss:
    return YoloDetectionLoss(
        DEFAULT_ANCHORS,
        num_classes=args.num_classes,
        image_size=args.image_size,
        box_weight=1.0,
        obj_weight=1.0,
        noobj_weight=0.25,
        cls_weight=0.0,
    )


def build_eval_image_transform(image_size: int, resize_mode: str) -> transforms.Compose:
    if resize_mode == "letterbox":
        resize_transform: object = ResizeWithPad(image_size)
    elif resize_mode == "stretch":
        resize_transform = transforms.Resize((image_size, image_size))
    else:
        raise ValueError(f"Unsupported resize_mode: {resize_mode}")
    return transforms.Compose(
        [
            resize_transform,
            transforms.ToTensor(),
        ]
    )


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    det_loss: YoloDetectionLoss,
    device: torch.device,
    class_names: Sequence[str],
    label_smoothing: float,
    grad_clip_norm: float = 0.0,
    limit_batches: int = 0,
) -> Dict[str, float]:
    model.train(True)
    num_classes = len(class_names)
    total_loss = 0.0
    total_det_loss = 0.0
    total_cls_loss = 0.0
    total_box_loss = 0.0
    total_obj_loss = 0.0
    total_samples = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for batch_index, (images, targets, _) in enumerate(loader):
        if limit_batches and batch_index >= limit_batches:
            break

        images = images.to(device)
        outputs = model(images)
        det_metrics = det_loss(outputs["det_preds"], targets)
        cls_targets = image_level_labels(targets, device=device)
        cls_loss = torch.nn.functional.cross_entropy(outputs["logits"], cls_targets, label_smoothing=label_smoothing)
        loss = det_metrics["loss"] + cls_loss

        optimizer.zero_grad()
        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip_norm)
        optimizer.step()

        probs = torch.softmax(outputs["logits"].detach(), dim=1)
        preds = probs.argmax(dim=1).cpu()
        targets_cpu = cls_targets.detach().cpu()
        cm += confusion_matrix(preds, targets_cpu, num_classes)

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_det_loss += float(det_metrics["loss"].item()) * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_box_loss += float(det_metrics["loss_box"].item()) * batch_size
        total_obj_loss += float(det_metrics["loss_obj"].item()) * batch_size
        total_samples += batch_size

    metrics = summarize_confusion_matrix(cm)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["det_loss"] = total_det_loss / max(total_samples, 1)
    metrics["cls_loss"] = total_cls_loss / max(total_samples, 1)
    metrics["loss_box"] = total_box_loss / max(total_samples, 1)
    metrics["loss_obj"] = total_obj_loss / max(total_samples, 1)
    return metrics


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader,
    det_loss: YoloDetectionLoss,
    device: torch.device,
    class_names: Sequence[str],
    label_smoothing: float,
    limit_batches: int = 0,
) -> Tuple[Dict[str, float], Tensor, List[Dict[str, float]], List[Dict[str, object]]]:
    model.train(False)
    num_classes = len(class_names)
    total_loss = 0.0
    total_det_loss = 0.0
    total_cls_loss = 0.0
    total_box_loss = 0.0
    total_obj_loss = 0.0
    total_samples = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    rows: List[Dict[str, object]] = []

    for batch_index, (images, targets, metadata_rows) in enumerate(loader):
        if limit_batches and batch_index >= limit_batches:
            break

        images = images.to(device)
        outputs = model(images)
        det_metrics = det_loss(outputs["det_preds"], targets)
        cls_targets = image_level_labels(targets, device=device)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        cls_loss = torch.nn.functional.cross_entropy(logits, cls_targets, label_smoothing=label_smoothing)
        loss = det_metrics["loss"] + cls_loss

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_det_loss += float(det_metrics["loss"].item()) * batch_size
        total_cls_loss += cls_loss.item() * batch_size
        total_box_loss += float(det_metrics["loss_box"].item()) * batch_size
        total_obj_loss += float(det_metrics["loss_obj"].item()) * batch_size
        total_samples += batch_size

        preds_cpu = preds.detach().cpu()
        probs_cpu = probs.detach().cpu()
        targets_cpu = cls_targets.detach().cpu()
        cm += confusion_matrix(preds_cpu, targets_cpu, num_classes)

        for sample_index, meta_row in enumerate(metadata_rows):
            target_id = int(targets_cpu[sample_index].item())
            pred_id = int(preds_cpu[sample_index].item())
            row = dict(meta_row)
            row["target_id"] = target_id
            row["target_name"] = class_names[target_id]
            row["pred_id"] = pred_id
            row["pred_name"] = class_names[pred_id]
            row["confidence"] = float(probs_cpu[sample_index, pred_id].item())
            for class_index, class_name in enumerate(class_names):
                row[f"prob_{class_index}"] = float(probs_cpu[sample_index, class_index].item())
                row[f"prob_name_{class_name}"] = float(probs_cpu[sample_index, class_index].item())
            rows.append(row)

    metrics = summarize_confusion_matrix(cm)
    metrics["loss"] = total_loss / max(total_samples, 1)
    metrics["det_loss"] = total_det_loss / max(total_samples, 1)
    metrics["cls_loss"] = total_cls_loss / max(total_samples, 1)
    metrics["loss_box"] = total_box_loss / max(total_samples, 1)
    metrics["loss_obj"] = total_obj_loss / max(total_samples, 1)
    report = classwise_report(cm, class_names)
    return metrics, cm, report, rows


def save_split_reports(
    output_dir: Path,
    split_name: str,
    class_names: Sequence[str],
    metrics: Dict[str, float],
    matrix: Tensor,
    report: List[Dict[str, float]],
    rows: List[Dict[str, object]],
) -> None:
    split_dir = output_dir / split_name
    split_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix_csv(matrix, class_names, split_dir / "frame_confusion_matrix.csv")
    save_rows_csv(rows, split_dir / "frame_predictions.csv")
    save_json({"metrics": metrics, "class_report": report}, split_dir / "frame_report.json")

    if rows and "video_id" in rows[0]:
        video_rows = aggregate_predictions_by_key(rows, num_classes=len(class_names), key="video_id")
        video_targets = torch.tensor([int(row["target_id"]) for row in video_rows], dtype=torch.long)
        video_preds = torch.tensor([int(row["pred_id"]) for row in video_rows], dtype=torch.long)
        video_matrix = confusion_matrix(video_preds, video_targets, len(class_names))
        video_metrics = summarize_confusion_matrix(video_matrix)
        video_report = classwise_report(video_matrix, class_names)
        save_rows_csv(video_rows, split_dir / "video_predictions.csv")
        save_confusion_matrix_csv(video_matrix, class_names, split_dir / "video_confusion_matrix.csv")
        save_json({"metrics": video_metrics, "class_report": video_report}, split_dir / "video_report.json")


@torch.no_grad()
def save_cam_samples(
    model: nn.Module,
    rows: Sequence[Dict[str, object]],
    class_names: Sequence[str],
    image_size: int,
    output_dir: Path,
    device: torch.device,
    max_samples: int,
    resize_mode: str = "stretch",
) -> List[Path]:
    if max_samples <= 0:
        return []

    chosen_rows: List[Dict[str, object]] = []
    seen_videos: set[str] = set()
    for row in rows:
        video_id = str(row.get("video_id", row.get("frame_path", "")))
        if video_id in seen_videos:
            continue
        seen_videos.add(video_id)
        chosen_rows.append(row)
        if len(chosen_rows) >= max_samples:
            break

    if not chosen_rows:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    transform = build_eval_image_transform(image_size=image_size, resize_mode=resize_mode)

    saved_paths: List[Path] = []
    for index, row in enumerate(chosen_rows):
        image_path = Path(str(row["frame_path"]))
        raw_image = Image.open(image_path).convert("RGB")
        resized = ResizeWithPad(image_size)(raw_image) if resize_mode == "letterbox" else raw_image.resize((image_size, image_size))
        tensor = transform(raw_image).unsqueeze(0).to(device)
        outputs = model(tensor)
        class_index = torch.tensor([int(row["pred_id"])], device=device)
        cam = model.compute_cam(outputs["features"], class_indices=class_index, input_size=(image_size, image_size))
        file_name = (
            f"{index:02d}_true-{sanitize_filename(str(row['target_name']))}"
            f"_pred-{sanitize_filename(str(row['pred_name']))}"
            f"_subject-{sanitize_filename(str(row.get('subject_id', 'na')))}"
            f"_video-{sanitize_filename(str(row.get('video_id', image_path.stem)))}.jpg"
        )
        saved_paths.append(save_cam_overlay(resized, cam[0].cpu().numpy(), output_dir / file_name))
    return saved_paths


def main() -> None:
    args = parse_args()
    args = apply_paper_detector_preset(args)
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_loader = build_loader(
        images_dir=args.train_images,
        labels_dir=args.train_labels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        resize_mode=args.resize_mode,
        num_workers=args.num_workers,
        shuffle=True,
    )
    val_loader = build_loader(
        images_dir=args.val_images,
        labels_dir=args.val_labels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        resize_mode=args.resize_mode,
        num_workers=args.num_workers,
        shuffle=False,
    )
    test_loader = build_loader(
        images_dir=args.test_images,
        labels_dir=args.test_labels,
        image_size=args.image_size,
        batch_size=args.batch_size,
        resize_mode=args.resize_mode,
        num_workers=args.num_workers,
        shuffle=False,
    )
    if train_loader is None:
        raise SystemExit("Training data is required.")

    class_names = infer_class_names(args.train_images, args.num_classes, args.class_names)
    det_loss = create_detection_loss(args)
    model = AttentionYOLOv3Drowsiness(
        num_classes=args.num_classes,
        detection_num_classes=args.num_classes,
        enable_detection=True,
        pretrained_backbone=args.pretrained,
        pretrained_backbone_path=args.pretrained_backbone_path,
    ).to(device)
    pretrained_source = getattr(model, "pretrained_backbone_source", None)
    optimizer = create_optimizer(args, model)
    scheduler = create_scheduler(args, optimizer)

    save_json(
        {
            "train_images": str(args.train_images),
            "train_labels": str(args.train_labels),
            "val_images": str(args.val_images) if args.val_images else None,
            "val_labels": str(args.val_labels) if args.val_labels else None,
            "test_images": str(args.test_images) if args.test_images else None,
            "test_labels": str(args.test_labels) if args.test_labels else None,
            "class_names": list(class_names),
            "num_classes": args.num_classes,
            "pretrained": args.pretrained,
            "pretrained_backbone_path": str(args.pretrained_backbone_path) if args.pretrained_backbone_path else None,
            "resolved_pretrained_backbone_path": pretrained_source,
            "image_size": args.image_size,
            "resize_mode": args.resize_mode,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "scheduler": args.scheduler,
            "min_lr": args.min_lr,
            "lr_milestones": list(args.lr_milestones),
            "lr_gamma": args.lr_gamma,
            "selection_metric": args.selection_metric,
            "label_smoothing": args.label_smoothing,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "grad_clip_norm": args.grad_clip_norm,
            "nesterov": args.nesterov,
            "device": str(device),
            "paper_preset": args.paper_preset,
            "anchors": DEFAULT_ANCHORS,
        },
        args.output_dir / "run_config.json",
    )

    best_metric_value = float("-inf")
    best_epoch = 0
    stopped_early = False
    stop_epoch = 0
    epochs_without_improvement = 0
    best_path = args.output_dir / "detector_best.pt"
    history_rows: List[Dict[str, object]] = []

    print(f"class_names={class_names}")
    print(
        f"pretrained={args.pretrained} optimizer={args.optimizer} scheduler={args.scheduler} lr={args.lr} "
        f"weight_decay={args.weight_decay} resize_mode={args.resize_mode}"
    )
    if pretrained_source is not None:
        print(f"pretrained_backbone_source={pretrained_source}")
    print(f"training_samples={len(train_loader.dataset)}")
    if val_loader is not None:
        print(f"validation_samples={len(val_loader.dataset)}")
    if test_loader is not None:
        print(f"test_samples={len(test_loader.dataset)}")

    train_limit = resolve_limit(args.limit_train_batches, args.limit_batches)
    val_limit = resolve_limit(args.limit_val_batches, args.limit_batches)
    test_limit = resolve_limit(args.limit_test_batches, args.limit_batches)

    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            det_loss=det_loss,
            device=device,
            class_names=class_names,
            label_smoothing=args.label_smoothing,
            grad_clip_norm=args.grad_clip_norm,
            limit_batches=train_limit,
        )

        if val_loader is not None:
            val_metrics, _, _, _ = evaluate_loader(
                model=model,
                loader=val_loader,
                det_loss=det_loss,
                device=device,
                class_names=class_names,
                label_smoothing=args.label_smoothing,
                limit_batches=val_limit,
            )
            selection_metrics = val_metrics
            selection_prefix = "val"
        else:
            selection_metrics = train_metrics
            selection_prefix = "train"

        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_det_loss": train_metrics["det_loss"],
            "train_cls_loss": train_metrics["cls_loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        if val_loader is not None:
            history_row.update(
                {
                    "val_loss": selection_metrics["loss"],
                    "val_det_loss": selection_metrics["det_loss"],
                    "val_cls_loss": selection_metrics["cls_loss"],
                    "val_accuracy": selection_metrics["accuracy"],
                    "val_f1_macro": selection_metrics["f1_macro"],
                }
            )
        history_rows.append(history_row)

        status_line = (
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} "
            f"train_det={train_metrics['det_loss']:.4f} "
            f"train_cls={train_metrics['cls_loss']:.4f} "
            f"train_acc={train_metrics['accuracy']:.4f} "
        )
        if val_loader is not None:
            status_line += (
                f"val_loss={selection_metrics['loss']:.4f} "
                f"val_det={selection_metrics['det_loss']:.4f} "
                f"val_cls={selection_metrics['cls_loss']:.4f} "
                f"val_acc={selection_metrics['accuracy']:.4f} "
                f"val_f1={selection_metrics['f1_macro']:.4f} "
            )
        status_line += (
            f"lr={optimizer.param_groups[0]['lr']:.6f} "
            f"select_{selection_prefix}_{args.selection_metric}={selection_metrics[args.selection_metric]:.4f}"
        )
        print(status_line)

        current_metric_value = float(selection_metrics[args.selection_metric])
        if current_metric_value > best_metric_value + args.early_stopping_min_delta:
            best_metric_value = current_metric_value
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": list(class_names),
                    "num_classes": args.num_classes,
                    "image_size": args.image_size,
                    "anchors": DEFAULT_ANCHORS,
                    "best_metric_name": args.selection_metric,
                    "best_metric_value": best_metric_value,
                    "best_epoch": best_epoch,
                    "resize_mode": args.resize_mode,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        if scheduler is not None:
            scheduler.step()

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            stop_epoch = epoch
            print(
                f"early_stopping_triggered=1 stop_epoch={stop_epoch} "
                f"best_epoch={best_epoch} best_{args.selection_metric}={best_metric_value:.4f}"
            )
            break

    save_rows_csv(history_rows, args.output_dir / "history.csv")

    if not best_path.exists():
        torch.save(
            {
                "model_state": model.state_dict(),
                "class_names": list(class_names),
                "num_classes": args.num_classes,
                "image_size": args.image_size,
                "anchors": DEFAULT_ANCHORS,
                "best_metric_name": args.selection_metric,
                "best_metric_value": best_metric_value,
                "best_epoch": best_epoch,
                "resize_mode": args.resize_mode,
            },
            best_path,
        )

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    val_cam_paths: List[Path] = []
    test_cam_paths: List[Path] = []

    if val_loader is not None:
        val_metrics, val_matrix, val_report, val_rows = evaluate_loader(
            model=model,
            loader=val_loader,
            det_loss=det_loss,
            device=device,
            class_names=class_names,
            label_smoothing=args.label_smoothing,
            limit_batches=val_limit,
        )
        save_split_reports(args.output_dir, "val", class_names, val_metrics, val_matrix, val_report, val_rows)
        val_cam_paths = save_cam_samples(
            model=model,
            rows=val_rows,
            class_names=class_names,
            image_size=args.image_size,
            output_dir=args.output_dir / "cam_samples" / "val",
            device=device,
            max_samples=args.cam_samples,
            resize_mode=args.resize_mode,
        )

    if test_loader is not None:
        test_metrics, test_matrix, test_report, test_rows = evaluate_loader(
            model=model,
            loader=test_loader,
            det_loss=det_loss,
            device=device,
            class_names=class_names,
            label_smoothing=args.label_smoothing,
            limit_batches=test_limit,
        )
        save_split_reports(args.output_dir, "test", class_names, test_metrics, test_matrix, test_report, test_rows)
        test_cam_paths = save_cam_samples(
            model=model,
            rows=test_rows,
            class_names=class_names,
            image_size=args.image_size,
            output_dir=args.output_dir / "cam_samples" / "test",
            device=device,
            max_samples=args.cam_samples,
            resize_mode=args.resize_mode,
        )

    summary_payload = {
        "best_checkpoint": str(best_path),
        "best_metric_name": args.selection_metric,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "stop_epoch": stop_epoch if stopped_early else len(history_rows),
        "class_names": list(class_names),
        "val_cam_samples": [str(path) for path in val_cam_paths],
        "test_cam_samples": [str(path) for path in test_cam_paths],
    }
    save_json(summary_payload, args.output_dir / "run_summary.json")

    print(f"best_checkpoint={best_path}")
    if val_loader is not None:
        print(f"val_report={args.output_dir / 'val' / 'frame_report.json'}")
    if test_loader is not None:
        print(f"test_report={args.output_dir / 'test' / 'frame_report.json'}")


if __name__ == "__main__":
    main()
