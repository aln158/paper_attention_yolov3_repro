from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np


YUNET_MODEL_URL = "https://files.kde.org/digikam/facesengine/yunet/face_detection_yunet_2023mar.onnx"


def default_model_path() -> Path:
    # Keep the detector model on an ASCII-only path because OpenCV ONNX loading
    # can fail on Windows paths containing non-ASCII characters.
    return Path.home() / ".paper_attention_yolo_models" / "face_detection_yunet_2023mar.onnx"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate eye-region YOLO pseudo labels for UTA-RLDD manifests using YuNet face detection. "
            "The output layout is split/images/... and split/labels/... so it can feed train_detector.py."
        )
    )
    parser.add_argument("--protocol-root", type=Path, required=True, help="Protocol directory containing train.csv/val.csv/test.csv.")
    parser.add_argument("--output-root", type=Path, required=True, help="Output root for detector-ready pseudo-labeled data.")
    parser.add_argument("--splits", nargs="+", default=["train", "val", "test"], help="Manifest splits to process.")
    parser.add_argument("--model-path", type=Path, default=default_model_path(), help="Local YuNet ONNX path.")
    parser.add_argument("--score-threshold", type=float, default=0.1, help="YuNet score threshold.")
    parser.add_argument("--nms-threshold", type=float, default=0.3, help="YuNet NMS threshold.")
    parser.add_argument("--top-k", type=int, default=5000, help="YuNet top-k candidate boxes.")
    parser.add_argument(
        "--horizontal-margin",
        type=float,
        default=0.35,
        help="Margin added on both sides of the eye span, relative to inter-eye distance.",
    )
    parser.add_argument(
        "--top-margin",
        type=float,
        default=0.12,
        help="Margin above the eye line, relative to face height.",
    )
    parser.add_argument(
        "--bottom-margin",
        type=float,
        default=0.18,
        help="Margin below the eye line, relative to face height.",
    )
    parser.add_argument("--min-box-width", type=float, default=12.0, help="Reject boxes narrower than this many pixels.")
    parser.add_argument("--min-box-height", type=float, default=10.0, help="Reject boxes shorter than this many pixels.")
    parser.add_argument(
        "--link-mode",
        choices=("hardlink", "copy"),
        default="hardlink",
        help="How to place source frames into the detector dataset.",
    )
    parser.add_argument("--max-frames-per-split", type=int, default=0, help="Limit frames per split for smoke tests. 0 means all.")
    parser.add_argument("--overwrite", action="store_true", help="Delete output-root before writing.")
    return parser.parse_args()


def ensure_model(model_path: Path) -> Path:
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    if not model_path.exists() or model_path.stat().st_size < 1024:
        urlretrieve(YUNET_MODEL_URL, model_path)
    return model_path


def read_manifest_rows(manifest_path: Path) -> List[Dict[str, str]]:
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise FileNotFoundError(f"No rows found in manifest: {manifest_path}")
    return rows


def unicode_imread(image_path: Path):
    buffer = np.fromfile(str(image_path), dtype=np.uint8)
    if buffer.size == 0:
        return None
    return cv2.imdecode(buffer, cv2.IMREAD_COLOR)


def link_or_copy_file(source: Path, destination: Path, link_mode: str) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return

    if link_mode == "hardlink":
        try:
            os.link(str(source), str(destination))
            return
        except OSError:
            pass

    shutil.copy2(source, destination)


def select_best_face(faces: np.ndarray) -> Optional[np.ndarray]:
    if faces is None or len(faces) == 0:
        return None
    return max(faces, key=lambda row: float(row[2] * row[3] * row[-1]))


def clip_box(x1: float, y1: float, x2: float, y2: float, width: int, height: int) -> Tuple[float, float, float, float]:
    x1 = max(0.0, min(float(width - 1), x1))
    y1 = max(0.0, min(float(height - 1), y1))
    x2 = max(0.0, min(float(width - 1), x2))
    y2 = max(0.0, min(float(height - 1), y2))
    return x1, y1, x2, y2


def eye_box_from_face(
    face_row: np.ndarray,
    image_width: int,
    image_height: int,
    horizontal_margin: float,
    top_margin: float,
    bottom_margin: float,
) -> Tuple[float, float, float, float]:
    face_x, face_y, face_w, face_h = [float(value) for value in face_row[:4]]
    right_eye_x, right_eye_y, left_eye_x, left_eye_y = [float(value) for value in face_row[4:8]]

    eye_left = min(right_eye_x, left_eye_x)
    eye_right = max(right_eye_x, left_eye_x)
    eye_top = min(right_eye_y, left_eye_y)
    eye_bottom = max(right_eye_y, left_eye_y)
    inter_eye_distance = max(eye_right - eye_left, face_w * 0.15)

    x1 = eye_left - inter_eye_distance * horizontal_margin
    x2 = eye_right + inter_eye_distance * horizontal_margin
    y1 = eye_top - face_h * top_margin
    y2 = eye_bottom + face_h * bottom_margin

    x1 = max(x1, face_x)
    y1 = max(y1, face_y)
    x2 = min(x2, face_x + face_w)
    y2 = min(y2, face_y + face_h)
    return clip_box(x1, y1, x2, y2, image_width, image_height)


def validate_box(box: Tuple[float, float, float, float], min_box_width: float, min_box_height: float) -> bool:
    x1, y1, x2, y2 = box
    return (x2 - x1) >= min_box_width and (y2 - y1) >= min_box_height


def box_to_yolo_row(class_id: int, box: Tuple[float, float, float, float], image_width: int, image_height: int) -> str:
    x1, y1, x2, y2 = box
    center_x = ((x1 + x2) * 0.5) / float(image_width)
    center_y = ((y1 + y2) * 0.5) / float(image_height)
    width = (x2 - x1) / float(image_width)
    height = (y2 - y1) / float(image_height)
    return f"{class_id} {center_x:.6f} {center_y:.6f} {width:.6f} {height:.6f}\n"


def write_csv(rows: Sequence[Dict[str, object]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        output_path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def process_split(
    split: str,
    manifest_path: Path,
    output_root: Path,
    detector,
    args: argparse.Namespace,
) -> Dict[str, object]:
    rows = read_manifest_rows(manifest_path)
    if args.max_frames_per_split > 0:
        rows = rows[: args.max_frames_per_split]

    images_root = output_root / split / "images"
    labels_root = output_root / split / "labels"
    metadata_root = output_root / "metadata"

    success_rows: List[Dict[str, object]] = []
    failure_rows: List[Dict[str, object]] = []
    last_box_by_video: Dict[str, Tuple[float, float, float, float]] = {}
    last_score_by_video: Dict[str, float] = {}

    for index, row in enumerate(rows, start=1):
        frame_path = Path(row["frame_path"])
        video_id = row["video_id"]
        relative_path = Path(row["class_name"]) / row["video_id"] / row["frame_name"]
        output_image_path = images_root / relative_path
        output_label_path = labels_root / relative_path.with_suffix(".txt")

        image = unicode_imread(frame_path)
        if image is None:
            failure_rows.append(
                {
                    "split": split,
                    "frame_path": str(frame_path),
                    "class_name": row["class_name"],
                    "class_id": row["class_id"],
                    "video_id": row["video_id"],
                    "reason": "image_read_failed",
                }
            )
            continue

        detector.setInputSize((image.shape[1], image.shape[0]))
        _, faces = detector.detect(image)
        best_face = select_best_face(faces)
        if best_face is None:
            box = last_box_by_video.get(video_id)
            face_score = last_score_by_video.get(video_id, 0.0)
            box_source = "previous_frame_box" if box is not None else "detector"
            if box is None:
                failure_rows.append(
                    {
                        "split": split,
                        "frame_path": str(frame_path),
                        "class_name": row["class_name"],
                        "class_id": row["class_id"],
                        "video_id": row["video_id"],
                        "reason": "no_face_detected",
                    }
                )
                continue
        else:
            box = eye_box_from_face(
                best_face,
                image_width=image.shape[1],
                image_height=image.shape[0],
                horizontal_margin=args.horizontal_margin,
                top_margin=args.top_margin,
                bottom_margin=args.bottom_margin,
            )
            face_score = float(best_face[-1])
            box_source = "detector"

        if not validate_box(box, min_box_width=args.min_box_width, min_box_height=args.min_box_height):
            failure_rows.append(
                {
                    "split": split,
                    "frame_path": str(frame_path),
                    "class_name": row["class_name"],
                    "class_id": row["class_id"],
                    "video_id": row["video_id"],
                    "reason": "invalid_eye_box",
                }
            )
            continue

        last_box_by_video[video_id] = box
        last_score_by_video[video_id] = face_score

        link_or_copy_file(frame_path, output_image_path, link_mode=args.link_mode)
        output_label_path.parent.mkdir(parents=True, exist_ok=True)
        output_label_path.write_text(
            box_to_yolo_row(
                class_id=int(row["class_id"]),
                box=box,
                image_width=image.shape[1],
                image_height=image.shape[0],
            ),
            encoding="utf-8",
        )

        success_rows.append(
            {
                **row,
                "output_image_path": str(output_image_path),
                "output_label_path": str(output_label_path),
                "face_score": round(float(face_score), 6),
                "box_source": box_source,
                "eye_box_x1": round(float(box[0]), 3),
                "eye_box_y1": round(float(box[1]), 3),
                "eye_box_x2": round(float(box[2]), 3),
                "eye_box_y2": round(float(box[3]), 3),
            }
        )

        if index % 500 == 0:
            print(
                f"[{split}] processed={index} success={len(success_rows)} "
                f"failed={len(failure_rows)}"
            )

    write_csv(success_rows, metadata_root / f"{split}_success.csv")
    write_csv(failure_rows, metadata_root / f"{split}_failure.csv")

    unique_videos = {row["video_id"] for row in success_rows}
    class_counts: Dict[str, int] = {}
    for row in success_rows:
        class_counts[row["class_name"]] = class_counts.get(row["class_name"], 0) + 1

    summary = {
        "split": split,
        "manifest_path": str(manifest_path),
        "rows_seen": len(rows),
        "rows_success": len(success_rows),
        "rows_failed": len(failure_rows),
        "success_rate": (len(success_rows) / len(rows)) if rows else 0.0,
        "videos_success": len(unique_videos),
        "class_frame_counts": class_counts,
        "images_root": str(images_root),
        "labels_root": str(labels_root),
    }
    return summary


def main() -> None:
    args = parse_args()
    protocol_root = args.protocol_root.resolve()
    output_root = args.output_root.resolve()
    model_path = ensure_model(args.model_path.resolve())

    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    detector = cv2.FaceDetectorYN.create(str(model_path), "", (320, 240))
    detector.setScoreThreshold(args.score_threshold)
    detector.setNMSThreshold(args.nms_threshold)
    detector.setTopK(args.top_k)

    split_summaries: List[Dict[str, object]] = []
    for split in args.splits:
        manifest_path = protocol_root / f"{split}.csv"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found for split '{split}': {manifest_path}")
        summary = process_split(split, manifest_path, output_root, detector, args)
        split_summaries.append(summary)
        print(
            f"[done] split={split} success={summary['rows_success']} "
            f"failed={summary['rows_failed']} success_rate={summary['success_rate']:.4f}"
        )

    payload = {
        "protocol_root": str(protocol_root),
        "output_root": str(output_root),
        "model_path": str(model_path),
        "score_threshold": args.score_threshold,
        "nms_threshold": args.nms_threshold,
        "top_k": args.top_k,
        "horizontal_margin": args.horizontal_margin,
        "top_margin": args.top_margin,
        "bottom_margin": args.bottom_margin,
        "min_box_width": args.min_box_width,
        "min_box_height": args.min_box_height,
        "link_mode": args.link_mode,
        "splits": split_summaries,
    }
    (output_root / "summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"summary={output_root / 'summary.json'}")


if __name__ == "__main__":
    main()
