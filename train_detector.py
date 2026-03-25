from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader

from repro.data import YoloDetectionDataset, detection_collate_fn
from repro.losses import YoloDetectionLoss
from repro.model import AttentionYOLOv3Drowsiness
from repro.utils import DEFAULT_ANCHORS, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the full attention YOLOv3 detector.")
    parser.add_argument("--train-images", type=Path, required=True)
    parser.add_argument("--train-labels", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=416)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "outputs")
    parser.add_argument("--limit-batches", type=int, default=0)
    return parser.parse_args()


def image_level_labels(targets: List[torch.Tensor], device: torch.device) -> torch.Tensor:
    labels = []
    for target in targets:
        if target.numel() == 0:
            labels.append(0)
        else:
            labels.append(int(target[0, 0].item()))
    return torch.tensor(labels, dtype=torch.long, device=device)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)
    device = torch.device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    dataset = YoloDetectionDataset(
        images_dir=args.train_images,
        labels_dir=args.train_labels,
        image_size=args.image_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=detection_collate_fn,
    )

    model = AttentionYOLOv3Drowsiness(
        num_classes=args.num_classes,
        detection_num_classes=args.num_classes,
        enable_detection=True,
    ).to(device)
    det_loss = YoloDetectionLoss(DEFAULT_ANCHORS, num_classes=args.num_classes, image_size=args.image_size)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    ckpt_path = args.output_dir / "detector_last.pt"
    print(f"training_samples={len(dataset)}")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        samples_seen = 0

        for batch_index, (images, targets) in enumerate(loader):
            if args.limit_batches and batch_index >= args.limit_batches:
                break

            images = images.to(device)
            outputs = model(images)
            det_metrics = det_loss(outputs["det_preds"], targets)
            cls_targets = image_level_labels(targets, device=device)
            cls_loss = torch.nn.functional.cross_entropy(outputs["logits"], cls_targets)
            loss = det_metrics["loss"] + cls_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_size = images.shape[0]
            epoch_loss += loss.item() * batch_size
            samples_seen += batch_size

        print(f"epoch={epoch} loss={epoch_loss / max(samples_seen, 1):.4f}")

    torch.save(
        {
            "model_state": model.state_dict(),
            "num_classes": args.num_classes,
            "image_size": args.image_size,
            "anchors": DEFAULT_ANCHORS,
        },
        ckpt_path,
    )
    print(f"checkpoint={ckpt_path}")


if __name__ == "__main__":
    main()
