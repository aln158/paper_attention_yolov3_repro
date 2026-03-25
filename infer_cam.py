from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from repro.model import AttentionYOLOv3Drowsiness
from repro.utils import save_cam_overlay


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CAM inference for the attention drowsiness model.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--image-path", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-path", type=Path, default=Path(__file__).resolve().parent / "outputs" / "cam_overlay.jpg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = torch.load(args.checkpoint, map_location=args.device)
    class_names = payload.get("class_names", ["alert", "drowsy"])
    image_size = int(payload.get("image_size", 224))
    enable_detection_head = bool(payload.get("enable_detection_head", False))

    model = AttentionYOLOv3Drowsiness(
        num_classes=len(class_names),
        detection_num_classes=len(class_names),
        enable_detection=enable_detection_head,
    ).to(args.device)
    model.load_state_dict(payload["model_state"])
    model.eval()

    image = Image.open(args.image_path).convert("RGB")
    resized = image.resize((image_size, image_size))
    tensor = transforms.ToTensor()(resized).unsqueeze(0).to(args.device)

    with torch.no_grad():
        outputs = model(tensor)
        probs = torch.softmax(outputs["logits"], dim=1)
        class_index = probs.argmax(dim=1)
        class_name = class_names[int(class_index.item())]
        confidence = float(probs[0, class_index].item())
        cam = model.compute_cam(outputs["features"], class_indices=class_index, input_size=(image_size, image_size))

    output_path = save_cam_overlay(resized, cam[0].cpu().numpy(), args.output_path)
    print(f"predicted_class={class_name} confidence={confidence:.4f} cam_path={output_path}")


if __name__ == "__main__":
    main()
