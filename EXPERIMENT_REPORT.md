# Experiment Report

Date: `2026-03-27`

Project: `paper_attention_yolov3_repro`

Paper target: `An Attention Deep Learning Framework-Based Drowsiness Detection Model for Intelligent Transportation System`

## 1. Goal

This project aims to reproduce the paper-style attention drowsiness classifier and adapt it to `UTA-RLDD` under leakage-safe subject/video-level protocols.

Current focus:

- `Protocol A`: strict binary classification, `alert` vs `drowsy`
- `Protocol B`: three-class classification, `alert / low_vigilant / drowsy`
- classification-first reproduction
- later alignment of full paper-style detection/classification loss

## 2. Data and Protocols

### 2.1 Legacy prepared-frame protocol

Source:

- `protocols/utarldd_protocol_a`
- built from the existing prepared low-resolution frame folders already present in the workspace

Characteristics:

- frame resolution roughly `80x60`
- subject/video leakage checked at manifest level
- used in the first round of completed experiments

`Protocol A` summary:

| Split | Subjects | Videos | Frames |
|---|---:|---:|---:|
| Train | 36 | 73 | 8534 |
| Val | 12 | 23 | 2760 |
| Test | 12 | 25 | 2944 |

Reference:

- `protocols/utarldd_protocol_a/summary.json`

### 2.2 Optimized re-extracted video protocol

Source:

- original `utarldd_data` videos
- re-extracted with `prepare_utarldd_optimized.py`

Extraction settings:

- `label_mode=multiclass`
- `sampling_mode=uniform`
- `frames_per_video=180`
- `resize=320x240`
- `jpeg_quality=95`
- `val_fold=4`
- `test_fold=5`

Generated protocol root:

- `protocols_opt_uniform_320`

`Protocol A` summary:

| Split | Subjects | Videos | Frames |
|---|---:|---:|---:|
| Train | 36 | 73 | 13140 |
| Val | 12 | 23 | 4139 |
| Test | 12 | 25 | 4499 |

`Protocol B` summary:

| Split | Subjects | Videos | Frames |
|---|---:|---:|---:|
| Train | 36 | 109 | 19620 |
| Val | 12 | 34 | 6119 |
| Test | 12 | 37 | 6659 |

References:

- `protocols_opt_uniform_320/utarldd_protocol_a/summary.json`
- `protocols_opt_uniform_320/utarldd_protocol_b/summary.json`

## 3. Completed Experiments

All completed experiments below were run on the legacy prepared-frame protocol unless noted otherwise.

### 3.1 Paper-style classifier baselines

| Run | Model | Pretrained | Image | Batch | Epochs | Optimizer | LR | WD | Scheduler | Best Val | Test Frame | Test Video | Notes |
|---|---|---|---:|---:|---:|---|---:|---:|---|---|---|---|---|
| `outputs_protocol_a_224` | `paper_attention_yolo` | No | 224 | 8 | 5 | legacy default | `1e-3` | not fully logged | none | final log `val_f1=0.5402` | `acc=0.6029`, `f1=0.5292` | `acc=0.6000`, `f1=0.5040` | first baseline |
| `outputs_protocol_a_224_v2` | `paper_attention_yolo` | No | 224 | 8 | 5 | legacy default | `3e-4` | `1e-4` | none | `best_val_f1=0.5063` | `acc=0.6478`, `f1=0.6478` | `acc=0.6400`, `f1=0.6400` | best completed run so far |
| `outputs_protocol_a_416_v1` | `paper_attention_yolo` | No | 416 | 4 | 5 | legacy default | `3e-4` | `1e-4` | none | `best_val_f1=0.5920` | `acc=0.3743`, `f1=0.3705` | `acc=0.3600`, `f1=0.3506` | validation looked better, test collapsed |

Key references:

- `outputs_protocol_a_224_v2/test/frame_report.json`
- `outputs_protocol_a_224_v2/test/video_report.json`

### 3.2 Pretrained ResNet18 transfer baselines

| Run | Model | Pretrained | Image | Batch | Epochs | Optimizer | LR | WD | Scheduler | Freeze | Best Val | Test Frame | Test Video | Notes |
|---|---|---|---:|---:|---:|---|---:|---:|---|---:|---|---|---|---|
| `outputs_resnet18_pretrained_a_224_e30_b8` | `attention_resnet18` | Yes | 224 | 8 | 30 | AdamW | `1e-3` | `1e-4` | cosine | 2 | `best_val_f1=0.6103` | `acc=0.5027`, `f1=0.4795` | `acc=0.5200`, `f1=0.5000` | early overfitting |
| `outputs_resnet18_pretrained_a_224_e100_b16` | `attention_resnet18` | Yes | 224 | 16 | 100 | AdamW | `5e-4` | `1e-4` | cosine | 2 | `best_val_f1=0.6444` | `acc=0.5516`, `f1=0.5509` | `acc=0.6000`, `f1=0.5994` | best ResNet18 run |
| `outputs_resnet18_pretrained_a_224_e30_b8_v2` | `attention_resnet18` | Yes | 224 | 8 | 30 | AdamW | `5e-4` | `1e-4` | cosine | 2 | `best_val_f1=0.6157` | `acc=0.4939`, `f1=0.4364` | `acc=0.5200`, `f1=0.4485` | unstable generalization |

## 4. Current Best Result

Best completed result to date:

- Run: `outputs_protocol_a_opt_uniform320_paper_sgd_224_lr2e3`
- Dataset: optimized re-extracted `Protocol A`
- Model: `paper_attention_yolo`
- Setting: `224`, `batch_size=8`, `epochs=150`, `optimizer=SGD`, `lr=0.002`, `weight_decay=5e-4`, `momentum=0.9`, `scheduler=MultiStepLR(60,110,gamma=0.1)`
- Early stopping: stopped at epoch `29`, best checkpoint from epoch `19`
- Best validation frame-level: `accuracy=0.6721`, `f1_macro=0.6716`
- Test frame-level: `accuracy=0.6824`, `f1_macro=0.6824`
- Test video-level: `accuracy=0.7200`, `f1_macro=0.7200`

This is now stronger than the previous legacy baseline `outputs_protocol_a_224_v2`.

## 5. New-Data Experiments on Optimized Extraction

### 5.1 ResNet50 pretrained on optimized `Protocol A`

Run:

- `outputs_resnet50_pretrained_protocol_a_opt_uniform320_224_e100_b16`

Configuration:

- model: `attention_resnet50`
- pretrained: `True`
- image size: `224`
- batch size: `16`
- epochs: `100`
- optimizer: `AdamW`
- lr: `3e-4`
- weight decay: `1e-4`
- scheduler: `cosine`
- freeze backbone epochs: `5`

Dataset size:

- train `13140`
- val `4139`
- test `4499`

Observed console log:

| Epoch | Train Loss | Train Acc | Val Loss | Val Acc | Val F1 |
|---|---:|---:|---:|---:|---:|
| 1 | 0.3687 | 0.9127 | 0.8542 | 0.5064 | 0.5044 |
| 2 | 0.1520 | 0.9865 | 0.9869 | 0.5030 | 0.5030 |
| 3 | 0.0910 | 0.9914 | 1.0890 | 0.4863 | 0.4858 |
| 4 | 0.0625 | 0.9934 | 1.2137 | 0.4815 | 0.4810 |
| 5 | 0.0461 | 0.9952 | 1.2877 | 0.5028 | 0.5009 |
| 6 | 0.0267 | 0.9913 | 2.9262 | 0.4359 | 0.4252 |

Interpretation:

- strong overfitting from the first few epochs
- best observed validation performance happened at epoch `1`
- as of now this line is clearly weaker than the best completed legacy baseline

Artifacts currently present:

- `outputs_resnet50_pretrained_protocol_a_opt_uniform320_224_e100_b16/run_config.json`
- `outputs_resnet50_pretrained_protocol_a_opt_uniform320_224_e100_b16/classifier_best.pt`

No final `run_summary.json` or `frame_report.json` was exported for this run.

### 5.2 Paper-style SGD baseline on optimized `Protocol A`, high learning rate

Run:

- `outputs_protocol_a_opt_uniform320_paper_sgd_224`

Configuration:

- model: `paper_attention_yolo`
- pretrained: `False`
- image size: `224`
- batch size: `8`
- epochs: `150`
- optimizer: `SGD`
- lr: `1e-2`
- weight decay: `5e-4`
- momentum: `0.9`
- scheduler: `multistep`
- milestones: `60, 110`
- gamma: `0.1`

Observed behavior:

- training converged too aggressively
- validation became unstable and overfit early
- this run was used mainly to show that `lr=0.01` was too large for the current classification setup

### 5.3 Paper-style SGD baseline on optimized `Protocol A`, reduced learning rate

Run:

- `outputs_protocol_a_opt_uniform320_paper_sgd_224_lr2e3`

Configuration:

- model: `paper_attention_yolo`
- pretrained: `False`
- image size: `224`
- batch size: `8`
- epochs: `150`
- optimizer: `SGD`
- lr: `2e-3`
- weight decay: `5e-4`
- momentum: `0.9`
- scheduler: `multistep`
- milestones: `60, 110`
- gamma: `0.1`
- early stopping patience: `10`
- selection metric: `f1_macro`

Best training summary:

| Item | Value |
|---|---:|
| Best epoch | `19` |
| Stop epoch | `29` |
| Best val accuracy | `0.6721` |
| Best val f1_macro | `0.6716` |

Saved evaluation:

| Split | Accuracy | Precision Macro | Recall Macro | F1 Macro | Loss |
|---|---:|---:|---:|---:|---:|
| Val frame | `0.6721` | `0.6775` | `0.6752` | `0.6716` | `1.5205` |
| Test frame | `0.6824` | `0.6834` | `0.6834` | `0.6824` | `1.7969` |
| Test video | `0.7200` | `0.7212` | `0.7212` | `0.7200` | - |

Test per-class frame metrics:

| Class | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| Alert | `0.6563` | `0.7101` | `0.6821` | `2159` |
| Drowsy | `0.7106` | `0.6568` | `0.6827` | `2340` |

Interpretation:

- this is the strongest completed run so far
- compared with `lr=0.01`, reducing the learning rate to `0.002` materially improved stability and generalization
- overfitting still starts after the best epoch, but early stopping preserved the best checkpoint

## 6. Implementation Updates During This Phase

### 6.1 Data pipeline

Added:

- `prepare_utarldd_optimized.py`

Purpose:

- re-extract frames directly from `UTA-RLDD` videos
- use whole-video uniform sampling
- keep better resolution than the old `80x60` prepared frames

### 6.2 Loss alignment with the paper

Updated:

- `repro/losses.py`
- `train_detector.py`

Main changes:

- detection localization loss moved closer to paper-style `MSE`
- objectness loss separated into positive-anchor and negative-anchor parts
- removed effective double-counting of classification loss inside the detector path

### 6.3 Optimizer and scheduler alignment

Updated:

- `train_classifier.py`
- `train_detector.py`

Added support for:

- `SGD`
- `MultiStepLR`
- paper-style milestones `60, 110`
- paper-style defaults `momentum=0.9`, `weight_decay=5e-4`

## 7. Summary of Findings

1. The current best completed result now comes from the custom `paper_attention_yolo` baseline on the optimized re-extracted `Protocol A`, trained with `SGD + multistep + lr=0.002`.
2. Increasing image size from `224` to `416` did not help on the legacy prepared-frame protocol.
3. Pretrained `ResNet18` improved optimization on the training split but did not beat the strongest custom baseline on test performance.
4. On the optimized higher-resolution uniformly extracted frames, pretrained `ResNet50` still showed very early overfitting.
5. Reducing the learning rate from `0.01` to `0.002` was the most effective change made so far on the optimized protocol.

## 8. Recommended Next Step

Use `outputs_protocol_a_opt_uniform320_paper_sgd_224_lr2e3` as the new main baseline and branch from it for the next comparison.

Most meaningful next experiments:

- add `pretrained` support only if the backbone is changed to a standard ImageNet model family
- keep the optimized extraction and protocol fixed while testing regularization or temporal aggregation
- compare against a clean three-class run on `Protocol B` only after locking the binary baseline

At this point the main bottleneck is no longer raw extraction quality alone; it is more likely related to label ambiguity, frame-level supervision mismatch, and missing temporal modeling.
