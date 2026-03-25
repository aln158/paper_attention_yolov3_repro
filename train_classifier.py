from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor, nn
from torchvision import transforms

from repro.data import (
    build_classification_loaders,
    build_manifest_loaders,
    discover_classification_samples,
)
from repro.model import AttentionResNetClassifier, AttentionYOLOv3Drowsiness
from repro.utils import (
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


def default_data_root() -> Optional[Path]:
    candidate = Path(__file__).resolve().parent.parent / "prepared_utarldd_mc_real" / "train"
    return candidate if candidate.exists() else None


def default_protocol_root() -> Path:
    return Path(__file__).resolve().parent / "protocols"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the attention drowsiness classifier.")
    parser.add_argument("--data-root", type=Path, default=default_data_root())
    parser.add_argument("--classes", nargs="+", default=["alert", "drowsy"])
    parser.add_argument(
        "--model-name",
        choices=("paper_attention_yolo", "attention_resnet18", "attention_resnet50"),
        default="paper_attention_yolo",
        help="Classifier architecture to train.",
    )
    parser.add_argument("--pretrained", action="store_true", help="Use torchvision ImageNet pretrained weights when supported.")
    parser.add_argument("--protocol", default="utarldd_protocol_a", help="Protocol directory under ./protocols")
    parser.add_argument("--train-manifest", type=Path, default=None)
    parser.add_argument("--val-manifest", type=Path, default=None)
    parser.add_argument("--test-manifest", type=Path, default=None)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--optimizer", choices=("adam", "adamw", "sgd"), default="adam")
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--scheduler", choices=("none", "cosine"), default="none")
    parser.add_argument("--min-lr", type=float, default=1e-6)
    parser.add_argument(
        "--freeze-backbone-epochs",
        type=int,
        default=0,
        help="Freeze supported backbones for the first N epochs.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=("accuracy", "f1_macro"),
        default="f1_macro",
        help="Validation metric used to select the best checkpoint.",
    )
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=2,
        help="Stop after this many epochs without validation improvement. Set 0 to disable.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=1e-4,
        help="Minimum validation improvement required to reset early stopping.",
    )
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--limit-train-batches", type=int, default=0)
    parser.add_argument("--limit-val-batches", type=int, default=0)
    parser.add_argument("--limit-test-batches", type=int, default=0)
    parser.add_argument("--enable-detection-head", action="store_true")
    parser.add_argument("--cam-samples", type=int, default=8)
    return parser.parse_args()


def resolve_normalization(model_name: str) -> Tuple[Optional[Sequence[float]], Optional[Sequence[float]]]:
    if model_name in {"attention_resnet18", "attention_resnet50"}:
        return (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
    return None, None


def create_classifier(args: argparse.Namespace, num_classes: int) -> nn.Module:
    if args.model_name == "paper_attention_yolo":
        return AttentionYOLOv3Drowsiness(
            num_classes=num_classes,
            detection_num_classes=num_classes,
            enable_detection=args.enable_detection_head,
        )
    if args.model_name == "attention_resnet18":
        return AttentionResNetClassifier(
            num_classes=num_classes,
            backbone_name="resnet18",
            pretrained=args.pretrained,
        )
    if args.model_name == "attention_resnet50":
        return AttentionResNetClassifier(
            num_classes=num_classes,
            backbone_name="resnet50",
            pretrained=args.pretrained,
        )
    raise ValueError(f"Unsupported model_name: {args.model_name}")


def create_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    trainable_parameters = list(model.parameters())
    if args.optimizer == "adam":
        return torch.optim.Adam(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(trainable_parameters, lr=args.lr, weight_decay=args.weight_decay)
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            trainable_parameters,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=True,
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
    raise ValueError(f"Unsupported scheduler: {args.scheduler}")


def maybe_set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    if hasattr(model, "set_backbone_trainable"):
        model.set_backbone_trainable(trainable)


def unpack_batch(batch) -> Tuple[Tensor, Tensor, Dict[str, List[str]]]:
    if len(batch) == 3:
        images, targets, metadata = batch
    elif len(batch) == 2:
        images, targets = batch
        metadata = {}
    else:
        raise ValueError(f"Unexpected batch structure: {type(batch)}")
    return images, targets, metadata


def collated_metadata_to_rows(metadata: Dict[str, Sequence[object]], batch_size: int) -> List[Dict[str, object]]:
    if not metadata:
        return [{} for _ in range(batch_size)]

    rows: List[Dict[str, object]] = []
    for index in range(batch_size):
        row: Dict[str, object] = {}
        for key, value in metadata.items():
            if isinstance(value, torch.Tensor):
                item = value[index]
                row[key] = item.item() if item.ndim == 0 else item.tolist()
            else:
                row[key] = value[index]
        rows.append(row)
    return rows


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    num_classes: int,
    label_smoothing: float,
    limit_batches: int = 0,
) -> Dict[str, float]:
    model.train(True)
    total_loss = 0.0
    total_samples = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)

    for batch_index, batch in enumerate(loader):
        if limit_batches and batch_index >= limit_batches:
            break

        images, targets, _ = unpack_batch(batch)
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        logits = outputs["logits"]
        loss = torch.nn.functional.cross_entropy(logits, targets, label_smoothing=label_smoothing)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        preds = logits.argmax(dim=1).detach().cpu()
        cm += confusion_matrix(preds, targets.detach().cpu(), num_classes)

    metrics = summarize_confusion_matrix(cm)
    metrics["loss"] = total_loss / max(total_samples, 1)
    return metrics


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader,
    device: torch.device,
    class_names: Sequence[str],
    label_smoothing: float,
    limit_batches: int = 0,
) -> Tuple[Dict[str, float], Tensor, List[Dict[str, float]], List[Dict[str, object]]]:
    model.train(False)
    num_classes = len(class_names)
    total_loss = 0.0
    total_samples = 0
    cm = torch.zeros((num_classes, num_classes), dtype=torch.long)
    rows: List[Dict[str, object]] = []

    for batch_index, batch in enumerate(loader):
        if limit_batches and batch_index >= limit_batches:
            break

        images, targets, metadata = unpack_batch(batch)
        images = images.to(device)
        targets = targets.to(device)
        outputs = model(images)
        logits = outputs["logits"]
        probs = torch.softmax(logits, dim=1)
        preds = probs.argmax(dim=1)
        loss = torch.nn.functional.cross_entropy(logits, targets, label_smoothing=label_smoothing)

        batch_size = images.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        preds_cpu = preds.detach().cpu()
        probs_cpu = probs.detach().cpu()
        targets_cpu = targets.detach().cpu()
        cm += confusion_matrix(preds_cpu, targets_cpu, num_classes)

        meta_rows = collated_metadata_to_rows(metadata, batch_size)
        for sample_index, meta_row in enumerate(meta_rows):
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
    report = classwise_report(cm, class_names)
    return metrics, cm, report, rows


def sanitize_filename(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", value)


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
    transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
        ]
    )

    saved_paths: List[Path] = []
    for index, row in enumerate(chosen_rows):
        image_path = Path(str(row["frame_path"]))
        raw_image = Image.open(image_path).convert("RGB")
        resized = raw_image.resize((image_size, image_size))
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


def resolve_manifest_paths(args: argparse.Namespace) -> Tuple[Optional[Path], Optional[Path], Optional[Path]]:
    if args.train_manifest and args.val_manifest:
        return args.train_manifest, args.val_manifest, args.test_manifest

    protocol_dir = default_protocol_root() / args.protocol
    train_manifest = protocol_dir / "train.csv"
    val_manifest = protocol_dir / "val.csv"
    test_manifest = protocol_dir / "test.csv"
    if train_manifest.exists() and val_manifest.exists():
        return train_manifest, val_manifest, test_manifest if test_manifest.exists() else None
    return None, None, None


def save_preview(
    model: nn.Module,
    device: torch.device,
    data_root: Path,
    class_names: Sequence[str],
    image_size: int,
    output_dir: Path,
    normalize_mean: Optional[Sequence[float]] = None,
    normalize_std: Optional[Sequence[float]] = None,
) -> Optional[Path]:
    samples = discover_classification_samples(data_root, class_names)
    sample_path, _ = samples[0]
    raw_image = Image.open(sample_path).convert("RGB")
    transform_steps: List[object] = [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ]
    if normalize_mean is not None and normalize_std is not None:
        transform_steps.append(transforms.Normalize(mean=list(normalize_mean), std=list(normalize_std)))
    transform = transforms.Compose(transform_steps)
    tensor = transform(raw_image).unsqueeze(0).to(device)
    outputs = model(tensor)
    class_indices = outputs["logits"].argmax(dim=1)
    cam = model.compute_cam(outputs["features"], class_indices=class_indices, input_size=(image_size, image_size))
    return save_cam_overlay(raw_image.resize((image_size, image_size)), cam[0].cpu().numpy(), output_dir / "preview_cam.jpg")


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    normalize_mean, normalize_std = resolve_normalization(args.model_name)

    train_manifest, val_manifest, test_manifest = resolve_manifest_paths(args)
    source_mode = "folder"
    if train_manifest is not None and val_manifest is not None:
        train_loader, val_loader, test_loader, class_names = build_manifest_loaders(
            train_manifest=train_manifest,
            val_manifest=val_manifest,
            test_manifest=test_manifest,
            image_size=args.image_size,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        source_mode = "manifest"
    else:
        if args.data_root is None:
            raise SystemExit("No dataset path found. Pass manifests or --data-root.")
        if not args.data_root.exists():
            raise SystemExit(f"Dataset root does not exist: {args.data_root}")
        train_loader, val_loader = build_classification_loaders(
            dataset_root=args.data_root,
            class_names=args.classes,
            image_size=args.image_size,
            batch_size=args.batch_size,
            val_ratio=args.val_ratio,
            seed=args.seed,
            num_workers=args.num_workers,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        test_loader = None
        class_names = list(args.classes)

    save_json(
        {
            "source_mode": source_mode,
            "protocol": args.protocol,
            "train_manifest": str(train_manifest) if train_manifest else None,
            "val_manifest": str(val_manifest) if val_manifest else None,
            "test_manifest": str(test_manifest) if test_manifest else None,
            "classes": list(class_names),
            "model_name": args.model_name,
            "pretrained": args.pretrained,
            "image_size": args.image_size,
            "batch_size": args.batch_size,
            "epochs": args.epochs,
            "lr": args.lr,
            "weight_decay": args.weight_decay,
            "optimizer": args.optimizer,
            "momentum": args.momentum,
            "scheduler": args.scheduler,
            "min_lr": args.min_lr,
            "freeze_backbone_epochs": args.freeze_backbone_epochs,
            "selection_metric": args.selection_metric,
            "label_smoothing": args.label_smoothing,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "normalize_mean": list(normalize_mean) if normalize_mean is not None else None,
            "normalize_std": list(normalize_std) if normalize_std is not None else None,
            "seed": args.seed,
            "device": str(device),
        },
        args.output_dir / "run_config.json",
    )

    model = create_classifier(args, num_classes=len(class_names)).to(device)
    maybe_set_backbone_trainable(model, trainable=args.freeze_backbone_epochs == 0)
    optimizer = create_optimizer(args, model)
    scheduler = create_scheduler(args, optimizer)
    best_metric_value = float("-inf")
    best_epoch = 0
    stopped_early = False
    stop_epoch = 0
    epochs_without_improvement = 0
    best_path = args.output_dir / "classifier_best.pt"
    history_rows: List[Dict[str, object]] = []

    print(f"source_mode={source_mode} classes={class_names}")
    print(f"model_name={args.model_name} pretrained={args.pretrained} optimizer={args.optimizer} scheduler={args.scheduler}")
    print(f"training_samples={len(train_loader.dataset)} validation_samples={len(val_loader.dataset)}")
    if test_loader is not None:
        print(f"test_samples={len(test_loader.dataset)}")

    for epoch in range(1, args.epochs + 1):
        maybe_set_backbone_trainable(model, trainable=epoch > args.freeze_backbone_epochs)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            num_classes=len(class_names),
            label_smoothing=args.label_smoothing,
            limit_batches=args.limit_train_batches,
        )
        val_metrics, _, _, _ = evaluate_loader(
            model,
            val_loader,
            device,
            class_names=class_names,
            label_smoothing=args.label_smoothing,
            limit_batches=args.limit_val_batches,
        )

        history_row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "train_f1_macro": train_metrics["f1_macro"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_f1_macro": val_metrics["f1_macro"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history_rows.append(history_row)
        print(
            f"epoch={epoch} "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['accuracy']:.4f} "
            f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
            f"val_f1={val_metrics['f1_macro']:.4f} "
            f"lr={optimizer.param_groups[0]['lr']:.6f} "
            f"select_{args.selection_metric}={val_metrics[args.selection_metric]:.4f}"
        )

        current_metric_value = float(val_metrics[args.selection_metric])
        if current_metric_value > best_metric_value + args.early_stopping_min_delta:
            best_metric_value = current_metric_value
            best_epoch = epoch
            epochs_without_improvement = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "class_names": list(class_names),
                    "image_size": args.image_size,
                    "enable_detection_head": args.enable_detection_head,
                    "model_name": args.model_name,
                    "pretrained": args.pretrained,
                    "best_metric_name": args.selection_metric,
                    "best_metric_value": best_metric_value,
                    "best_epoch": best_epoch,
                    "source_mode": source_mode,
                },
                best_path,
            )
        else:
            epochs_without_improvement += 1

        if args.early_stopping_patience > 0 and epochs_without_improvement >= args.early_stopping_patience:
            stopped_early = True
            stop_epoch = epoch
            print(
                f"early_stopping_triggered=1 stop_epoch={stop_epoch} "
                f"best_epoch={best_epoch} best_{args.selection_metric}={best_metric_value:.4f}"
            )
            break

        if scheduler is not None:
            scheduler.step()

    save_rows_csv(history_rows, args.output_dir / "history.csv")

    checkpoint = torch.load(best_path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    val_metrics, val_matrix, val_report, val_rows = evaluate_loader(
        model,
        val_loader,
        device,
        class_names=class_names,
        label_smoothing=args.label_smoothing,
        limit_batches=args.limit_val_batches,
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
    )

    if test_loader is not None:
        test_metrics, test_matrix, test_report, test_rows = evaluate_loader(
            model,
            test_loader,
            device,
            class_names=class_names,
            label_smoothing=args.label_smoothing,
            limit_batches=args.limit_test_batches,
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
        )
    else:
        test_cam_paths = []

    if source_mode == "folder":
        preview_path = save_preview(
            model=model,
            device=device,
            data_root=args.data_root,
            class_names=class_names,
            image_size=args.image_size,
            output_dir=args.output_dir,
            normalize_mean=normalize_mean,
            normalize_std=normalize_std,
        )
        if preview_path is not None:
            print(f"preview_cam={preview_path}")

    summary_payload = {
        "best_checkpoint": str(best_path),
        "model_name": args.model_name,
        "pretrained": args.pretrained,
        "best_metric_name": args.selection_metric,
        "best_metric_value": best_metric_value,
        "best_epoch": best_epoch,
        "stopped_early": stopped_early,
        "stop_epoch": stop_epoch if stopped_early else len(history_rows),
        "val_cam_samples": [str(path) for path in val_cam_paths],
        "test_cam_samples": [str(path) for path in test_cam_paths],
        "class_names": list(class_names),
    }
    save_json(summary_payload, args.output_dir / "run_summary.json")

    print(f"best_checkpoint={best_path}")
    print(f"val_report={args.output_dir / 'val' / 'frame_report.json'}")
    if test_loader is not None:
        print(f"test_report={args.output_dir / 'test' / 'frame_report.json'}")


if __name__ == "__main__":
    main()
