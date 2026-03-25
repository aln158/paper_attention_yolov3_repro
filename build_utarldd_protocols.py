from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Sequence


PROTOCOLS = {
    "utarldd_protocol_a": {
        "description": "Binary protocol: alert vs drowsy.",
        "classes": ["alert", "drowsy"],
    },
    "utarldd_protocol_b": {
        "description": "Three-class protocol: alert vs low_vigilant vs drowsy.",
        "classes": ["alert", "low_vigilant", "drowsy"],
    },
}

VIDEO_DIR_PATTERN = re.compile(
    r"^(?P<fold_name>Fold(?P<fold>\d+)_part(?P<part>\d+))__(?P=fold_name)__(?P<subject>[^_]+)__(?P<clip>.+)$",
    re.IGNORECASE,
)


def default_source_root() -> Path:
    return Path(__file__).resolve().parent.parent / "AD4DBR-main" / "codes on 100-Driver" / "data" / "utarldd" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build protocol manifests for UTA-RLDD.")
    parser.add_argument("--source-root", type=Path, default=default_source_root())
    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "protocols")
    return parser.parse_args()


def parse_video_dir_name(video_dir_name: str) -> Dict[str, str]:
    match = VIDEO_DIR_PATTERN.match(video_dir_name)
    if not match:
        raise ValueError(f"Unexpected video directory format: {video_dir_name}")
    return {
        "fold_name": match.group("fold_name"),
        "fold": match.group("fold"),
        "part": match.group("part"),
        "subject_id": match.group("subject"),
        "clip_id": match.group("clip"),
    }


def iter_frame_rows(source_root: Path, split: str, class_names: Sequence[str]) -> Iterable[Dict[str, str]]:
    for class_id, class_name in enumerate(class_names):
        class_dir = source_root / split / class_name
        if not class_dir.exists():
            continue

        for video_dir in sorted(path for path in class_dir.iterdir() if path.is_dir()):
            meta = parse_video_dir_name(video_dir.name)
            frame_paths = sorted(video_dir.glob("*.jpg"))
            for frame_index, frame_path in enumerate(frame_paths):
                yield {
                    "split": split,
                    "class_name": class_name,
                    "class_id": str(class_id),
                    "fold_name": meta["fold_name"],
                    "fold": meta["fold"],
                    "part": meta["part"],
                    "subject_id": meta["subject_id"],
                    "clip_id": meta["clip_id"],
                    "video_id": video_dir.name,
                    "video_dir": str(video_dir.resolve()),
                    "frame_index": str(frame_index),
                    "frame_name": frame_path.name,
                    "frame_path": str(frame_path.resolve()),
                }


def summarize_rows(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    summary: Dict[str, object] = {"splits": {}}
    grouped_split_class_frames: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    grouped_split_class_videos: Dict[str, Dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    grouped_split_class_subjects: Dict[str, Dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    split_subjects: Dict[str, set[str]] = defaultdict(set)
    split_videos: Dict[str, set[str]] = defaultdict(set)

    for row in rows:
        split = row["split"]
        class_name = row["class_name"]
        grouped_split_class_frames[split][class_name] += 1
        grouped_split_class_videos[split][class_name].add(row["video_id"])
        grouped_split_class_subjects[split][class_name].add(row["subject_id"])
        split_subjects[split].add(row["subject_id"])
        split_videos[split].add(row["video_id"])

    for split in sorted(split_subjects):
        summary["splits"][split] = {
            "subjects": len(split_subjects[split]),
            "videos": len(split_videos[split]),
            "frames": sum(grouped_split_class_frames[split].values()),
            "classes": {},
        }
        for class_name in sorted(grouped_split_class_frames[split]):
            summary["splits"][split]["classes"][class_name] = {
                "subjects": len(grouped_split_class_subjects[split][class_name]),
                "videos": len(grouped_split_class_videos[split][class_name]),
                "frames": grouped_split_class_frames[split][class_name],
            }

    return summary


def validate_no_leakage(rows: Sequence[Dict[str, str]]) -> Dict[str, object]:
    split_subjects: Dict[str, set[str]] = defaultdict(set)
    split_videos: Dict[str, set[str]] = defaultdict(set)
    for row in rows:
        split_subjects[row["split"]].add(row["subject_id"])
        split_videos[row["split"]].add(row["video_id"])

    splits = sorted(split_subjects)
    subject_overlap: Dict[str, List[str]] = {}
    video_overlap: Dict[str, List[str]] = {}
    for index, left in enumerate(splits):
        for right in splits[index + 1 :]:
            subject_key = f"{left}__{right}"
            video_key = f"{left}__{right}"
            subject_overlap[subject_key] = sorted(split_subjects[left] & split_subjects[right])
            video_overlap[video_key] = sorted(split_videos[left] & split_videos[right])
    return {
        "subject_overlap": subject_overlap,
        "video_overlap": video_overlap,
    }


def write_manifest(rows: Sequence[Dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "split",
        "class_name",
        "class_id",
        "fold_name",
        "fold",
        "part",
        "subject_id",
        "clip_id",
        "video_id",
        "video_dir",
        "frame_index",
        "frame_name",
        "frame_path",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not source_root.exists():
        raise SystemExit(f"Source root does not exist: {source_root}")

    for protocol_name, protocol_meta in PROTOCOLS.items():
        protocol_dir = output_root / protocol_name
        protocol_dir.mkdir(parents=True, exist_ok=True)

        all_rows: List[Dict[str, str]] = []
        for split in ("train", "val", "test"):
            rows = list(iter_frame_rows(source_root, split=split, class_names=protocol_meta["classes"]))
            write_manifest(rows, protocol_dir / f"{split}.csv")
            all_rows.extend(rows)
            print(f"{protocol_name} {split}: frames={len(rows)}")

        summary = summarize_rows(all_rows)
        leakage = validate_no_leakage(all_rows)
        payload = {
            "protocol_name": protocol_name,
            "description": protocol_meta["description"],
            "source_root": str(source_root),
            "summary": summary,
            "leakage_check": leakage,
        }
        (protocol_dir / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"{protocol_name}: wrote manifests to {protocol_dir}")


if __name__ == "__main__":
    main()
