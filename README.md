# Attention YOLOv3 drowsiness reproduction

This folder contains a practical PyTorch reproduction of the paper:

`An Attention Deep Learning Framework-Based Drowsiness Detection Model for Intelligent Transportation System`

The reproduction is designed for the current workspace:

- Full model structure is implemented.
- A classification training path is adapted to the local `prepared_utarldd_mc_real/train` dataset.
- A detector training path is included for YOLO-format datasets when box labels are available.

## Files

- `MODEL_PLAN.md`: architecture and reproduction notes
- `UTARLDD_PROTOCOLS.md`: phase 1/2 protocol details and commands
- `build_utarldd_protocols.py`: generate leakage-safe manifest files for protocol A and B
- `prepare_utarldd_eye_pseudolabels.py`: generate detector-ready eye-region pseudo labels from UTARLDD manifests
- `train_classifier.py`: train the attention backbone and classification head on folder-based image labels
- `train_detector.py`: train the full YOLO-style branch on YOLO-format labels
- `infer_cam.py`: run inference and save a CAM overlay
- `repro/model.py`: model definition
- `repro/data.py`: dataset loaders
- `repro/losses.py`: YOLO-style detection loss
- `repro/utils.py`: metrics, decoding, CAM overlay, seed helpers

## Expected local classification dataset layout

The classifier script expects a folder layout like:

```text
prepared_utarldd_mc_real/train/
  alert/
    any_subdir/
      *.jpg
  drowsy/
    any_subdir/
      *.jpg
  low_vigilant/
    any_subdir/
      *.jpg
```

For a paper-faithful two-class setting, pass:

```text
--classes alert drowsy
```

## Quick start

Generate UTARLDD protocol manifests:

```powershell
& C:\Users\allen\.conda\envs\pytorch\python.exe .\build_utarldd_protocols.py
```

Tensor-shape smoke test:

```powershell
& C:\Users\allen\.conda\envs\pytorch\python.exe .\train_classifier.py --epochs 1 --batch-size 4 --image-size 160 --limit-train-batches 1 --limit-val-batches 1
```

Train protocol A on leakage-safe UTARLDD manifests:

```powershell
& C:\Users\allen\.conda\envs\pytorch\python.exe .\train_classifier.py --protocol utarldd_protocol_a --epochs 5 --batch-size 8 --image-size 224 --lr 3e-4 --weight-decay 1e-4 --selection-metric f1_macro --early-stopping-patience 2
```

Train a stronger pretrained transfer-learning baseline:

```powershell
& C:\Users\allen\.conda\envs\pytorch\python.exe .\train_classifier.py --model-name attention_resnet18 --pretrained --protocol utarldd_protocol_a --epochs 20 --batch-size 8 --image-size 224 --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --scheduler cosine --min-lr 1e-5 --freeze-backbone-epochs 2 --selection-metric f1_macro --early-stopping-patience 4 --output-dir .\outputs_resnet18_pretrained_a_224
```

Generate a CAM overlay:

```powershell
& C:\Users\allen\.conda\envs\pytorch\python.exe .\infer_cam.py --checkpoint .\outputs\classifier_best.pt --image-path C:\path\to\sample.jpg
```

Train detector branch on YOLO labels:

```powershell
& C:\Users\allen\.conda\envs\pytorch\python.exe .\train_detector.py --train-images C:\dataset\images\train --train-labels C:\dataset\labels\train
```

Generate eye-region pseudo labels for UTARLDD protocol A:

```powershell
& C:\Users\allen\.conda\envs\pytorch\python.exe .\prepare_utarldd_eye_pseudolabels.py --protocol-root .\protocols_opt_uniform_320\utarldd_protocol_a --output-root .\detector_pseudolabels_protocol_a --overwrite
```

Run the paper-style detector experiment with val/test export:

```powershell
& C:\Users\allen\.conda\envs\pytorch\python.exe .\train_detector.py `
  --train-images '.\detector_pseudolabels_protocol_a\train\images' `
  --train-labels '.\detector_pseudolabels_protocol_a\train\labels' `
  --val-images '.\detector_pseudolabels_protocol_a\val\images' `
  --val-labels '.\detector_pseudolabels_protocol_a\val\labels' `
  --test-images '.\detector_pseudolabels_protocol_a\test\images' `
  --test-labels '.\detector_pseudolabels_protocol_a\test\labels' `
  --class-names alert drowsy `
  --paper-preset `
  --selection-metric f1_macro `
  --output-dir '.\outputs_paper_strict_detector_protocol_a'
```

## Notes

- Paper input size is `416`, but smaller sizes are useful for CPU smoke tests.
- The local workspace uses `torch 2.8.0+cpu`, so quick verification is CPU-oriented.
- When only classification labels are available, the detector branch remains implemented but untrained.
- The pseudo-label script stores the YuNet ONNX model under `C:\Users\allen\.paper_attention_yolo_models` to avoid Windows non-ASCII path issues when OpenCV loads ONNX files.
