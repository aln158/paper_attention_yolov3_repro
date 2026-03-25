# UTARLDD protocols

This project now supports two leakage-safe UTARLDD protocols built from:

`AD4DBR-main/codes on 100-Driver/data/utarldd/images`

The split logic follows the existing fold-based organization already present in the workspace, so train, val, and test do not share subjects or videos.

## Protocol A

Binary classification:

- `alert`
- `drowsy`

Counts:

- train: `36` subjects, `73` videos, `8534` frames
- val: `12` subjects, `23` videos, `2760` frames
- test: `12` subjects, `25` videos, `2944` frames

Files:

- `protocols/utarldd_protocol_a/train.csv`
- `protocols/utarldd_protocol_a/val.csv`
- `protocols/utarldd_protocol_a/test.csv`
- `protocols/utarldd_protocol_a/summary.json`

## Protocol B

Three-class classification:

- `alert`
- `low_vigilant`
- `drowsy`

Counts:

- train: `36` subjects, `109` videos, `12797` frames
- val: `12` subjects, `34` videos, `4080` frames
- test: `12` subjects, `37` videos, `4384` frames

Files:

- `protocols/utarldd_protocol_b/train.csv`
- `protocols/utarldd_protocol_b/val.csv`
- `protocols/utarldd_protocol_b/test.csv`
- `protocols/utarldd_protocol_b/summary.json`

## Phase 1 command

Build or refresh both protocol manifests:

```powershell
& C:\Users\allen\.conda\envs\pytorch\python.exe .\build_utarldd_protocols.py
```

## Phase 2 commands

Protocol A debug run at `224`:

```powershell
& C:\Users\allen\.conda\envs\pytorch\python.exe .\train_classifier.py --protocol utarldd_protocol_a --epochs 5 --batch-size 8 --image-size 224 --lr 3e-4 --weight-decay 1e-4 --selection-metric f1_macro --early-stopping-patience 2 --output-dir .\outputs_protocol_a_224
```

Protocol A formal run at `416`:

```powershell
& C:\Users\allen\.conda\envs\pytorch\python.exe .\train_classifier.py --protocol utarldd_protocol_a --epochs 5 --batch-size 4 --image-size 416 --lr 3e-4 --weight-decay 1e-4 --selection-metric f1_macro --early-stopping-patience 2 --output-dir .\outputs_protocol_a_416
```

Protocol B debug run at `224`:

```powershell
& C:\Users\allen\.conda\envs\pytorch\python.exe .\train_classifier.py --protocol utarldd_protocol_b --epochs 5 --batch-size 8 --image-size 224 --lr 3e-4 --weight-decay 1e-4 --selection-metric f1_macro --early-stopping-patience 2 --output-dir .\outputs_protocol_b_224
```

Practical high-accuracy attempt with pretrained ResNet18 and cosine schedule:

```powershell
& C:\Users\allen\.conda\envs\pytorch\python.exe .\train_classifier.py --model-name attention_resnet18 --pretrained --protocol utarldd_protocol_a --epochs 20 --batch-size 8 --image-size 224 --optimizer adamw --lr 1e-3 --weight-decay 1e-4 --scheduler cosine --min-lr 1e-5 --freeze-backbone-epochs 2 --selection-metric f1_macro --early-stopping-patience 4 --output-dir .\outputs_resnet18_pretrained_a_224
```

Notes:

- The prepared UTARLDD frames in this workspace are low-resolution, around `80x60`.
- Because of that, `224` is usually a safer practical choice than `416` for transfer learning.
- `416` stays useful for paper-style reproduction, but not necessarily for best accuracy on this prepared dataset.

## Training outputs

Each run writes:

- `classifier_best.pt`
- `history.csv`
- `val/frame_report.json`
- `val/frame_confusion_matrix.csv`
- `val/video_report.json`
- `val/video_confusion_matrix.csv`
- `test/frame_report.json`
- `test/video_report.json`
- `cam_samples/val/*.jpg`
- `cam_samples/test/*.jpg`
