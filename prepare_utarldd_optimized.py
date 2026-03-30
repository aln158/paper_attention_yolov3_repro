from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import cv2


VIDEO_EXTENSIONS = {".avi", ".mp4", ".mov", ".mkv", ".mpeg", ".mpg"}
LABEL_PATTERNS = {
    "alert": [r"alert"],
    "low_vigilant": [r"low[\s_-]*vigilant", r"low[\s_-]*alert"],
    "drowsy": [r"drowsy", r"sleepy"],
}
NUMERIC_LABEL_MAP = {"0": "alert", "5": "low_vigilant", "10": "drowsy"}
FOLD_PATTERN = re.compile(r"fold[\s_-]*(\d+)", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Optimized UTA-RLDD frame extraction for the paper_attention_yolov3_repro pipeline. "
            "Outputs split/class/video_dir/frame.jpg folders compatible with build_utarldd_protocols.py."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True, help="Root directory of extracted UTA-RLDD videos.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for extracted frames.")
    parser.add_argument(
        "--label-mode",
        choices=("binary", "multiclass"),
        default="multiclass",
        help="binary => alert/drowsy, multiclass => alert/low_vigilant/drowsy.",
    )
    parser.add_argument(
        "--sampling-mode",
        choices=("uniform", "tail_uniform", "time_step"),
        default="uniform",
        help="uniform covers the whole video evenly; tail_uniform samples evenly from the last part of the video; time_step mimics fixed-interval extraction.",
    )
    parser.add_argument(
        "--frames-per-video",
        type=int,
        default=180,
        help="Maximum number of frames to save per video. Recommended 120-180 for current single-frame experiments.",
    )
    parser.add_argument(
        "--frame-step-seconds",
        type=float,
        default=1.0,
        help="Used only when sampling-mode=time_step.",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.5,
        help="Used only when sampling-mode=tail_uniform. 0.5 means sample from the last half of each video.",
    )
    parser.add_argument("--resize-width", type=int, default=320, help="Output image width. Use 0 to keep original size.")
    parser.add_argument("--resize-height", type=int, default=240, help="Output image height. Use 0 to keep original size.")
    parser.add_argument("--jpeg-quality", type=int, default=95, help="JPEG quality from 1 to 100.")
    parser.add_argument("--val-fold", type=int, default=4, help="Fold reserved for validation.")
    parser.add_argument("--test-fold", type=int, default=5, help="Fold reserved for testing.")
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Fallback train ratio when fold folders are not present.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Fallback validation ratio when fold folders are not present.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip videos whose output folder already contains jpg files.",
    )
    return parser.parse_args()


def find_videos(dataset_root: Path) -> List[Path]:
    return sorted(path for path in dataset_root.rglob("*") if path.suffix.lower() in VIDEO_EXTENSIONS)


def infer_label(path: Path) -> str | None:
    normalized = str(path).replace("\\", "/").lower()
    for label, patterns in LABEL_PATTERNS.items():
        if any(re.search(pattern, normalized) for pattern in patterns):
            return label

    stem = path.stem.lower()
    for numeric_prefix, label in NUMERIC_LABEL_MAP.items():
        if stem == numeric_prefix or stem.startswith(f"{numeric_prefix}_"):
            return label

    parent_name = path.parent.name.lower()
    if parent_name in NUMERIC_LABEL_MAP:
        return NUMERIC_LABEL_MAP[parent_name]
    return None


def remap_label(label: str, label_mode: str) -> str:
    if label_mode == "multiclass":
        return label
    return "drowsy" if label == "drowsy" else "alert"


def infer_split(path: Path, val_fold: int, test_fold: int, train_ratio: float, val_ratio: float) -> str:
    normalized = str(path).replace("\\", "/")
    match = FOLD_PATTERN.search(normalized)
    if match:
        fold = int(match.group(1))
        if fold == test_fold:
            return "test"
        if fold == val_fold:
            return "val"
        return "train"

    digest = hashlib.md5(normalized.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    if bucket < train_ratio:
        return "train"
    if bucket < train_ratio + val_ratio:
        return "val"
    return "test"


def make_unique_video_dir(dataset_root: Path, output_root: Path, video_path: Path, split: str, mapped_label: str) -> Path:
    relative_parent = video_path.parent.relative_to(dataset_root)
    safe_parts: List[str] = []
    for part in relative_parent.parts:
        safe_parts.append(re.sub(r"[^A-Za-z0-9._-]+", "_", part))
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", video_path.stem)
    unique_name = "__".join([*safe_parts, safe_stem])
    return output_root / split / mapped_label / unique_name


def unique_sorted(values: Sequence[int]) -> List[int]:
    seen: set[int] = set()
    output: List[int] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        output.append(value)
    return output


def sample_uniform_indices(frame_count: int, max_frames: int) -> List[int]:
    if frame_count <= 0:
        return []
    target = frame_count if max_frames <= 0 else min(frame_count, max_frames)
    if target <= 1:
        return [0]
    return unique_sorted([round(index * (frame_count - 1) / (target - 1)) for index in range(target)])


def sample_tail_uniform_indices(frame_count: int, max_frames: int, tail_fraction: float) -> List[int]:
    if frame_count <= 0:
        return []
    clamped_fraction = min(max(tail_fraction, 1e-3), 1.0)
    start_index = int(round(frame_count * (1.0 - clamped_fraction)))
    start_index = min(max(start_index, 0), max(frame_count - 1, 0))
    tail_frame_count = frame_count - start_index
    if tail_frame_count <= 0:
        return sample_uniform_indices(frame_count, max_frames)
    tail_indices = sample_uniform_indices(tail_frame_count, max_frames)
    return [start_index + index for index in tail_indices]


def sample_time_step_indices(frame_count: int, fps: float, frame_step_seconds: float, max_frames: int) -> List[int]:
    if frame_count <= 0:
        return []
    if not fps or fps <= 0:
        fps = 25.0
    stride = max(1, int(round(fps * frame_step_seconds)))
    indices = list(range(0, frame_count, stride))
    if max_frames > 0:
        indices = indices[:max_frames]
    return indices


def resize_frame(frame, width: int, height: int):
    if width <= 0 or height <= 0:
        return frame
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)


def clean_destination_dir(destination_dir: Path) -> None:
    if destination_dir.exists():
        shutil.rmtree(destination_dir)
    destination_dir.mkdir(parents=True, exist_ok=True)


def extract_frames_for_video(
    video_path: Path,
    destination_dir: Path,
    sampling_mode: str,
    frames_per_video: int,
    frame_step_seconds: float,
    tail_fraction: float,
    resize_width: int,
    resize_height: int,
    jpeg_quality: int,
) -> int:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        print(f"[WARN] Failed to open: {video_path}")
        return 0

    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if sampling_mode == "uniform" and frame_count > 0:
        selected_indices = sample_uniform_indices(frame_count, frames_per_video)
    elif sampling_mode == "tail_uniform" and frame_count > 0:
        selected_indices = sample_tail_uniform_indices(frame_count, frames_per_video, tail_fraction)
    elif sampling_mode == "time_step" and frame_count > 0:
        selected_indices = sample_time_step_indices(frame_count, fps, frame_step_seconds, frames_per_video)
    else:
        if sampling_mode in {"uniform", "tail_uniform"}:
            print(f"[WARN] Unknown frame count for {video_path.name}; falling back to time_step sampling.")
        selected_indices = []

    clean_destination_dir(destination_dir)

    if selected_indices:
        wanted = set(selected_indices)
        written = 0
        frame_index = 0
        success, frame = capture.read()
        while success:
            if frame_index in wanted:
                resized = resize_frame(frame, resize_width, resize_height)
                output_path = destination_dir / f"{video_path.stem}_{written:04d}.jpg"
                encoded, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
                if encoded:
                    buffer.tofile(str(output_path))
                    written += 1
                    if written >= len(selected_indices):
                        break
            frame_index += 1
            success, frame = capture.read()
        capture.release()
        return written

    if not fps or fps <= 0:
        fps = 25.0
    stride = max(1, int(round(fps * frame_step_seconds)))
    written = 0
    frame_index = 0
    success, frame = capture.read()
    while success and (frames_per_video <= 0 or written < frames_per_video):
        if frame_index % stride == 0:
            resized = resize_frame(frame, resize_width, resize_height)
            output_path = destination_dir / f"{video_path.stem}_{written:04d}.jpg"
            encoded, buffer = cv2.imencode(".jpg", resized, [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)])
            if encoded:
                buffer.tofile(str(output_path))
                written += 1
        frame_index += 1
        success, frame = capture.read()
    capture.release()
    return written


def summarize_counts(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    split_summary: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: {"videos": 0, "frames": 0}))
    seen_videos: Dict[str, set[str]] = defaultdict(set)

    for row in rows:
        split = row["split"]
        label = row["label"]
        split_summary[split][label]["frames"] += int(row["frames"])
        video_key = row["video_dir"]
        if video_key not in seen_videos[split]:
            seen_videos[split].add(video_key)
            split_summary[split][label]["videos"] += 1

    payload: Dict[str, object] = {"splits": {}}
    for split in sorted(split_summary):
        classes = split_summary[split]
        payload["splits"][split] = {
            "videos": sum(item["videos"] for item in classes.values()),
            "frames": sum(item["frames"] for item in classes.values()),
            "classes": classes,
        }
    return payload


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    videos = find_videos(dataset_root)
    if not videos:
        raise SystemExit(f"No videos found under {dataset_root}")

    processed = 0
    skipped = 0
    total_frames = 0
    rows: List[Dict[str, str]] = []
    for video_path in videos:
        label = infer_label(video_path)
        if label is None:
            skipped += 1
            print(f"[SKIP] Could not infer label from path: {video_path}")
            continue

        split = infer_split(
            video_path,
            val_fold=args.val_fold,
            test_fold=args.test_fold,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
        )
        mapped_label = remap_label(label, args.label_mode)
        target_dir = make_unique_video_dir(dataset_root, output_root, video_path, split, mapped_label)

        if args.skip_existing and target_dir.exists() and any(target_dir.glob("*.jpg")):
            frames = len(list(target_dir.glob("*.jpg")))
            print(f"[SKIP] {video_path.name} -> split={split}, label={mapped_label}, existing_frames={frames}")
        else:
            frames = extract_frames_for_video(
                video_path=video_path,
                destination_dir=target_dir,
                sampling_mode=args.sampling_mode,
                frames_per_video=args.frames_per_video,
                frame_step_seconds=args.frame_step_seconds,
                tail_fraction=args.tail_fraction,
                resize_width=args.resize_width,
                resize_height=args.resize_height,
                jpeg_quality=args.jpeg_quality,
            )
            print(
                f"[OK] {video_path.name} -> split={split}, label={mapped_label}, "
                f"frames={frames}, sampling={args.sampling_mode}"
            )

        processed += 1
        total_frames += frames
        rows.append(
            {
                "video_name": video_path.name,
                "split": split,
                "label": mapped_label,
                "frames": str(frames),
                "video_dir": str(target_dir),
            }
        )

    summary = {
        "dataset_root": str(dataset_root),
        "output_root": str(output_root),
        "label_mode": args.label_mode,
        "sampling_mode": args.sampling_mode,
        "frames_per_video": args.frames_per_video,
        "frame_step_seconds": args.frame_step_seconds,
        "tail_fraction": args.tail_fraction,
        "resize_width": args.resize_width,
        "resize_height": args.resize_height,
        "jpeg_quality": args.jpeg_quality,
        "processed_videos": processed,
        "skipped_videos": skipped,
        "total_frames": total_frames,
        "summary": summarize_counts(rows),
    }
    summary_path = output_root / "extraction_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Finished. processed_videos={processed}, skipped_videos={skipped}, total_frames={total_frames}")
    print(f"summary_json={summary_path}")


if __name__ == "__main__":
    main()
