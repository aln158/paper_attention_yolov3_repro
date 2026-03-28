from __future__ import annotations

import csv
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}
VIDEO_DIR_PATTERN = re.compile(
    r"^(?P<fold_name>Fold(?P<fold>\d+)_part(?P<part>\d+))__(?P=fold_name)__(?P<subject>[^_]+)__(?P<clip>.+)$",
    re.IGNORECASE,
)


class ResizeWithPad:
    def __init__(self, image_size: int, fill: Tuple[int, int, int] = (0, 0, 0)) -> None:
        self.image_size = image_size
        self.fill = fill

    def __call__(self, image: Image.Image) -> Image.Image:
        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Invalid image size: {(width, height)}")

        scale = min(self.image_size / float(width), self.image_size / float(height))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = image.resize((resized_width, resized_height), Image.BILINEAR)

        canvas = Image.new(image.mode, (self.image_size, self.image_size), self.fill)
        pad_left = (self.image_size - resized_width) // 2
        pad_top = (self.image_size - resized_height) // 2
        canvas.paste(resized, (pad_left, pad_top))
        return canvas


def resize_yolo_sample_with_pad(image: Image.Image, labels: Tensor, image_size: int) -> Tuple[Image.Image, Tensor]:
    width, height = image.size
    if width <= 0 or height <= 0:
        raise ValueError(f"Invalid image size: {(width, height)}")

    scale = min(image_size / float(width), image_size / float(height))
    resized_width = max(1, int(round(width * scale)))
    resized_height = max(1, int(round(height * scale)))
    resized = image.resize((resized_width, resized_height), Image.BILINEAR)

    canvas = Image.new(image.mode, (image_size, image_size), (0, 0, 0))
    pad_left = (image_size - resized_width) // 2
    pad_top = (image_size - resized_height) // 2
    canvas.paste(resized, (pad_left, pad_top))

    if labels.numel() == 0:
        return canvas, labels

    updated = labels.clone()
    x_center = labels[:, 1] * float(width) * scale + float(pad_left)
    y_center = labels[:, 2] * float(height) * scale + float(pad_top)
    box_width = labels[:, 3] * float(width) * scale
    box_height = labels[:, 4] * float(height) * scale

    updated[:, 1] = x_center / float(image_size)
    updated[:, 2] = y_center / float(image_size)
    updated[:, 3] = box_width / float(image_size)
    updated[:, 4] = box_height / float(image_size)
    return canvas, updated


class ClassificationFolderDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[Path, int]],
        class_names: Sequence[str],
        transform: Optional[Callable] = None,
    ) -> None:
        self.samples = list(samples)
        self.class_names = list(class_names)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, int, Dict[str, str]]:
        path, label = self.samples[index]
        image = Image.open(path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        metadata = {
            "frame_path": str(path),
            "video_id": path.parent.name,
            "subject_id": path.parent.name,
            "split": path.parts[-4] if len(path.parts) >= 4 else "unknown",
        }
        return image, label, metadata


class ManifestClassificationDataset(Dataset):
    def __init__(self, rows: Sequence[Dict[str, str]], transform: Optional[Callable] = None) -> None:
        self.rows = list(rows)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> Tuple[Tensor, int, Dict[str, str]]:
        row = self.rows[index]
        image = Image.open(row["frame_path"]).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        metadata = dict(row)
        return image, int(row["class_id"]), metadata


class YoloDetectionDataset(Dataset):
    def __init__(
        self,
        images_dir: Path,
        labels_dir: Path,
        image_size: int,
        resize_mode: str = "stretch",
        transform: Optional[Callable] = None,
    ) -> None:
        self.images_dir = Path(images_dir)
        self.labels_dir = Path(labels_dir)
        self.image_size = image_size
        self.resize_mode = resize_mode
        self.transform = transform
        self.samples = self._discover_samples()

    def _discover_samples(self) -> List[Tuple[Path, Path]]:
        items: List[Tuple[Path, Path]] = []
        for image_path in sorted(self.images_dir.rglob("*")):
            if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            rel_path = image_path.relative_to(self.images_dir).with_suffix(".txt")
            label_path = self.labels_dir / rel_path
            items.append((image_path, label_path))
        return items

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Tensor, Tensor, Dict[str, str]]:
        image_path, label_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        labels = read_yolo_label_file(label_path)
        metadata = self._build_metadata(image_path=image_path, label_path=label_path)
        if self.transform is not None:
            image = self.transform(image)
            return image, labels, metadata

        if self.resize_mode == "letterbox":
            image, labels = resize_yolo_sample_with_pad(image, labels, self.image_size)
        elif self.resize_mode == "stretch":
            image = image.resize((self.image_size, self.image_size), Image.BILINEAR)
        else:
            raise ValueError(f"Unsupported resize_mode: {self.resize_mode}")

        image = TF.to_tensor(image)
        return image, labels, metadata

    def _build_metadata(self, image_path: Path, label_path: Path) -> Dict[str, str]:
        relative_path = image_path.relative_to(self.images_dir)
        class_name = relative_path.parts[0] if len(relative_path.parts) >= 1 else "unknown"
        video_id = relative_path.parts[1] if len(relative_path.parts) >= 2 else image_path.parent.name
        subject_id = video_id
        fold_name = ""
        fold = ""
        part = ""
        clip_id = ""

        match = VIDEO_DIR_PATTERN.match(video_id)
        if match:
            subject_id = match.group("subject")
            fold_name = match.group("fold_name")
            fold = match.group("fold")
            part = match.group("part")
            clip_id = match.group("clip")

        return {
            "frame_path": str(image_path),
            "label_path": str(label_path),
            "frame_name": image_path.name,
            "video_id": video_id,
            "subject_id": subject_id,
            "class_name": class_name,
            "split": self.images_dir.parent.name,
            "fold_name": fold_name,
            "fold": fold,
            "part": part,
            "clip_id": clip_id,
        }


def read_yolo_label_file(label_path: Path) -> Tensor:
    if not label_path.exists():
        return torch.zeros((0, 5), dtype=torch.float32)

    rows: List[List[float]] = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) != 5:
            raise ValueError(f"Invalid YOLO row in {label_path}: {line}")
        rows.append([float(part) for part in parts])
    return torch.tensor(rows, dtype=torch.float32) if rows else torch.zeros((0, 5), dtype=torch.float32)


def discover_classification_samples(dataset_root: Path, class_names: Sequence[str]) -> List[Tuple[Path, int]]:
    dataset_root = Path(dataset_root)
    samples: List[Tuple[Path, int]] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = dataset_root / class_name
        if not class_dir.exists():
            continue
        for path in sorted(class_dir.rglob("*")):
            if path.suffix.lower() in IMAGE_EXTENSIONS:
                samples.append((path, class_index))
    if not samples:
        raise FileNotFoundError(f"No images found under {dataset_root} for classes {list(class_names)}")
    return samples


def stratified_split(
    samples: Sequence[Tuple[Path, int]],
    val_ratio: float,
    seed: int,
) -> Tuple[List[Tuple[Path, int]], List[Tuple[Path, int]]]:
    grouped: Dict[int, List[Tuple[Path, int]]] = defaultdict(list)
    for sample in samples:
        grouped[sample[1]].append(sample)

    rng = random.Random(seed)
    train_samples: List[Tuple[Path, int]] = []
    val_samples: List[Tuple[Path, int]] = []

    for group in grouped.values():
        shuffled = list(group)
        rng.shuffle(shuffled)
        val_count = int(round(len(shuffled) * val_ratio))
        if len(shuffled) > 1:
            val_count = min(max(val_count, 1), len(shuffled) - 1)
        val_samples.extend(shuffled[:val_count])
        train_samples.extend(shuffled[val_count:])

    rng.shuffle(train_samples)
    rng.shuffle(val_samples)
    return train_samples, val_samples


def read_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    manifest_path = Path(manifest_path)
    rows: List[Dict[str, str]] = []
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            row = {key: value for key, value in row.items() if key is not None}
            if not row:
                continue
            frame_path = row.get("frame_path")
            if not frame_path:
                raise ValueError(f"Manifest row missing frame_path in {manifest_path}")
            row["frame_path"] = str(Path(frame_path))
            rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No rows found in manifest: {manifest_path}")
    return rows


def infer_class_names_from_rows(rows: Sequence[Dict[str, str]]) -> List[str]:
    id_to_name: Dict[int, str] = {}
    for row in rows:
        id_to_name[int(row["class_id"])] = row["class_name"]
    return [name for _, name in sorted(id_to_name.items())]


def build_classification_transforms(
    image_size: int,
    normalize_mean: Optional[Sequence[float]] = None,
    normalize_std: Optional[Sequence[float]] = None,
    resize_mode: str = "stretch",
    augmentation_policy: str = "default",
) -> Tuple[Callable, Callable]:
    if resize_mode == "letterbox":
        resize_transform: Callable = ResizeWithPad(image_size)
    elif resize_mode == "stretch":
        resize_transform = transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BILINEAR)
    else:
        raise ValueError(f"Unsupported resize_mode: {resize_mode}")

    if augmentation_policy == "paper":
        train_steps: List[Callable] = [
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(
                degrees=10,
                translate=(0.05, 0.05),
                scale=(0.9, 1.1),
                interpolation=InterpolationMode.BILINEAR,
                fill=0,
            ),
            resize_transform,
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
        ]
    elif augmentation_policy == "none":
        train_steps = [
            resize_transform,
            transforms.ToTensor(),
        ]
    elif augmentation_policy == "default":
        train_steps = [
            resize_transform,
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
        ]
    else:
        raise ValueError(f"Unsupported augmentation_policy: {augmentation_policy}")

    eval_steps: List[Callable] = [
        resize_transform,
        transforms.ToTensor(),
    ]
    if normalize_mean is not None and normalize_std is not None:
        normalize = transforms.Normalize(mean=list(normalize_mean), std=list(normalize_std))
        train_steps.append(normalize)
        eval_steps.append(normalize)

    train_transform = transforms.Compose(train_steps)
    eval_transform = transforms.Compose(eval_steps)
    return train_transform, eval_transform


def build_classification_loaders(
    dataset_root: Path,
    class_names: Sequence[str],
    image_size: int,
    batch_size: int,
    val_ratio: float = 0.2,
    seed: int = 42,
    num_workers: int = 0,
    normalize_mean: Optional[Sequence[float]] = None,
    normalize_std: Optional[Sequence[float]] = None,
    resize_mode: str = "stretch",
    augmentation_policy: str = "default",
) -> Tuple[DataLoader, DataLoader]:
    samples = discover_classification_samples(Path(dataset_root), class_names)
    train_samples, val_samples = stratified_split(samples, val_ratio=val_ratio, seed=seed)
    train_transform, eval_transform = build_classification_transforms(
        image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        resize_mode=resize_mode,
        augmentation_policy=augmentation_policy,
    )

    train_dataset = ClassificationFolderDataset(train_samples, class_names=class_names, transform=train_transform)
    val_dataset = ClassificationFolderDataset(val_samples, class_names=class_names, transform=eval_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    return train_loader, val_loader


def build_manifest_loaders(
    train_manifest: Path,
    val_manifest: Path,
    image_size: int,
    batch_size: int,
    num_workers: int = 0,
    test_manifest: Optional[Path] = None,
    normalize_mean: Optional[Sequence[float]] = None,
    normalize_std: Optional[Sequence[float]] = None,
    resize_mode: str = "stretch",
    augmentation_policy: str = "default",
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], List[str]]:
    train_rows = read_manifest_rows(train_manifest)
    val_rows = read_manifest_rows(val_manifest)
    test_rows = read_manifest_rows(test_manifest) if test_manifest is not None else None
    class_names = infer_class_names_from_rows(train_rows)

    train_transform, eval_transform = build_classification_transforms(
        image_size,
        normalize_mean=normalize_mean,
        normalize_std=normalize_std,
        resize_mode=resize_mode,
        augmentation_policy=augmentation_policy,
    )
    train_dataset = ManifestClassificationDataset(train_rows, transform=train_transform)
    val_dataset = ManifestClassificationDataset(val_rows, transform=eval_transform)
    test_dataset = ManifestClassificationDataset(test_rows, transform=eval_transform) if test_rows is not None else None

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=False,
    )
    test_loader = None
    if test_dataset is not None:
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=False,
        )
    return train_loader, val_loader, test_loader, class_names


def detection_collate_fn(
    batch: Iterable[Tuple[Tensor, Tensor, Dict[str, str]]]
) -> Tuple[Tensor, List[Tensor], List[Dict[str, str]]]:
    images, labels, metadata = zip(*batch)
    return torch.stack(list(images), dim=0), list(labels), list(metadata)
