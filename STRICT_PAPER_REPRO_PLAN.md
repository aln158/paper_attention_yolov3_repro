# Strict Paper Reproduction Plan

Project: `paper_attention_yolov3_repro`

Target paper:
`An Attention Deep Learning Framework-Based Drowsiness Detection Model for Intelligent Transportation System`

## 1. What "strict paper reproduction" means here

To count as a strict reproduction, the experiment should follow the paper in these aspects:

- model: `paper_attention_yolo`
- backbone: Darknet-53 style backbone with the paper attention module
- no pretrained weights
- input size: `416 x 416`
- optimizer: `SGD`
- weight decay: `5e-4`
- momentum: `0.9`
- epochs: `150`
- learning-rate schedule:
  - epoch `1-60`: `0.01`
  - epoch `61-110`: `0.001`
  - epoch `111-150`: `0.0001`
- batch size: `40`
- loss: full paper-style objective
  - `L = Lloc + Lobj + Lclass`
- task: eye-region detection and eye-state classification jointly
- output: detection results, classification metrics, and CAM interpretation

## 2. What is already aligned in this repo

Code paths:

- `repro/model.py`
- `repro/losses.py`
- `train_detector.py`
- `train_classifier.py`

Already available:

- paper-style `paper_attention_yolo` backbone
- attention module with CAM path
- three-scale detector heads
- paper-style SGD training support
- paper-style multistep schedule support
- paper-style `416` input support
- letterbox resize support
- paper-style augmentation preset for the classification approximation
- paper-style preset flag:
  - `train_classifier.py --paper-preset`
  - `train_detector.py --paper-preset`

## 3. What is still missing for a truly strict reproduction

The current UTARLDD / RLDD pipeline is still missing YOLO eye bounding-box annotations.

That means:

- `train_classifier.py` can only do a classification-only approximation
- `train_detector.py` cannot be used properly until we have eye-region labels

So the main blocker is not the backbone or optimizer anymore.
The blocker is detection supervision.

## 4. Recommended strict-reproduction workflow on UTARLDD

### Phase A. Lock the binary label protocol

Use strict two-class labels:

- `alert`
- `drowsy`

Do not merge `low_vigilant` into `alert` for the strict binary paper-style run.

Use the existing leakage-safe split:

- train by subject/video
- val by subject/video
- test by subject/video

Current manifest root:

- `protocols_opt_uniform_320/utarldd_protocol_a`

### Phase B. Generate eye-region detection labels

This is the key missing step.

Recommended practical route:

1. Detect face and landmarks on each extracted frame.
2. Estimate left-eye and right-eye regions from landmarks.
3. Merge them into one eye-region box, or keep a single union box around the eye band.
4. Export YOLO labels:
   - `class_id x_center y_center width height`
5. Review a subset manually.
6. Drop low-confidence or obviously failed detections.

Best practical source for pseudo-label generation:

- facial landmark detector or face-alignment model
- then convert landmarks to an eye-region box

Why this is the best compromise:

- much faster than full manual box annotation
- much closer to the paper than whole-frame classification
- lets us train the actual detector branch

### Phase C. Build detector-ready split folders

Prepare folders like:

```text
detector_data/
  train/
    images/
    labels/
  val/
    images/
    labels/
  test/
    images/
    labels/
```

Important:

- the image split must match the existing subject/video-safe protocol
- labels must stay paired with the same split

### Phase D. Run the strict detector training

When the YOLO labels are ready, use:

```powershell
& C:\Users\allen\.conda\envs\pytorch\python.exe .\train_detector.py `
  --train-images '.\detector_data\train\images' `
  --train-labels '.\detector_data\train\labels' `
  --num-classes 2 `
  --paper-preset `
  --output-dir '.\outputs_paper_strict_detector_train'
```

Current `--paper-preset` in `train_detector.py` sets:

- `image_size=416`
- `resize_mode=letterbox`
- `optimizer=sgd`
- `lr=0.01`
- `weight_decay=5e-4`
- `momentum=0.9`
- `scheduler=multistep`
- `lr_milestones=60 110`
- `lr_gamma=0.1`
- `epochs=150`
- `batch_size=40`

## 5. What to run meanwhile

Before detector labels are ready, the closest available approximation is:

```powershell
& C:\Users\allen\.conda\envs\pytorch\python.exe .\train_classifier.py `
  --paper-preset `
  --train-manifest '.\protocols_opt_uniform_320\utarldd_protocol_a\train.csv' `
  --val-manifest '.\protocols_opt_uniform_320\utarldd_protocol_a\val.csv' `
  --test-manifest '.\protocols_opt_uniform_320\utarldd_protocol_a\test.csv' `
  --output-dir '.\outputs_protocol_a_paper_strict_classifier_approx'
```

This is useful as a control baseline, but it is still not the full paper setup.

## 6. Suggested next milestone

The next milestone should be:

- generate eye-region pseudo-box labels for UTARLDD

Once that is done, we can move the project from:

- classification-only approximation

to:

- actual paper-style detector training

That is the point where the reproduction becomes genuinely close to the original paper rather than only structurally similar.
