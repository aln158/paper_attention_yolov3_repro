# AutoDL Runbook

This project can run on AutoDL with a CUDA-enabled PyTorch image.

## 1. Recommended instance setup

- Use `1 x NVIDIA GPU` with at least `16 GB` VRAM for the detector at `416`.
- A `24 GB` card is a safer choice if you want to try larger batch sizes or heavier backbones.
- Choose a `PyTorch` image that already includes `CUDA` and `Python 3.10+`.
- If available, prefer an instance with at least `4-8 CPU cores` for one GPU.

## 2. Upload code and data

Recommended structure on the server:

```bash
/root/project/paper_attention_yolov3_repro
/root/data/UTARLDD
```

You can either:

- `git clone` the repo on AutoDL
- upload a zip in JupyterLab and unzip it
- use VSCode Remote-SSH to open the repo directly

## 3. Enter the instance

Open a terminal in JupyterLab or connect by SSH / VSCode Remote-SSH.

## 4. Prepare the environment

If the selected image already has GPU PyTorch installed, do **not** reinstall `torch` unless you intentionally want a different CUDA build.

Run:

```bash
cd /root/project/paper_attention_yolov3_repro
python -V
nvidia-smi
python -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available())"
python -m pip install --upgrade pip
python -m pip install numpy opencv-python-headless Pillow
```

Expected result:

- `nvidia-smi` shows your rented GPU
- `torch.cuda.is_available()` prints `True`

## 5. Long-running jobs

For long training runs, use a persistent terminal and save logs:

```bash
apt-get update && apt-get install -y screen
screen -S drowsy
```

Inside `screen`, run training with:

```bash
python your_script.py > train.log 2>&1
tail -f train.log
```

Detach from `screen` with:

```text
Ctrl+A, then D
```

Reattach later with:

```bash
screen -r drowsy
```

## 6. Classifier training on GPU

Example:

```bash
cd /root/project/paper_attention_yolov3_repro
python train_classifier.py \
  --model-name attention_resnet18 \
  --pretrained \
  --train-manifest /root/project/paper_attention_yolov3_repro/protocols_opt_uniform_320/utarldd_protocol_a/train.csv \
  --val-manifest /root/project/paper_attention_yolov3_repro/protocols_opt_uniform_320/utarldd_protocol_a/val.csv \
  --test-manifest /root/project/paper_attention_yolov3_repro/protocols_opt_uniform_320/utarldd_protocol_a/test.csv \
  --epochs 30 \
  --batch-size 16 \
  --image-size 224 \
  --optimizer adamw \
  --lr 5e-4 \
  --weight-decay 1e-4 \
  --scheduler cosine \
  --min-lr 1e-5 \
  --freeze-backbone-epochs 2 \
  --selection-metric f1_macro \
  --early-stopping-patience 6 \
  --num-workers 4 \
  --device cuda \
  --output-dir /root/project/paper_attention_yolov3_repro/outputs_resnet18_pretrained_a_224_autodl
```

## 7. Pseudo-label generation

If you want the detector pipeline:

```bash
cd /root/project/paper_attention_yolov3_repro
python prepare_utarldd_eye_pseudolabels.py \
  --protocol-root /root/project/paper_attention_yolov3_repro/protocols_opt_uniform_320/utarldd_protocol_a \
  --output-root /root/project/paper_attention_yolov3_repro/detector_pseudolabels_protocol_a \
  --overwrite
```

## 8. Detector training on GPU

Stable detector command:

```bash
cd /root/project/paper_attention_yolov3_repro
python train_detector.py \
  --train-images /root/project/paper_attention_yolov3_repro/detector_pseudolabels_protocol_a/train/images \
  --train-labels /root/project/paper_attention_yolov3_repro/detector_pseudolabels_protocol_a/train/labels \
  --val-images /root/project/paper_attention_yolov3_repro/detector_pseudolabels_protocol_a/val/images \
  --val-labels /root/project/paper_attention_yolov3_repro/detector_pseudolabels_protocol_a/val/labels \
  --test-images /root/project/paper_attention_yolov3_repro/detector_pseudolabels_protocol_a/test/images \
  --test-labels /root/project/paper_attention_yolov3_repro/detector_pseudolabels_protocol_a/test/labels \
  --class-names alert drowsy \
  --epochs 100 \
  --batch-size 8 \
  --image-size 416 \
  --resize-mode letterbox \
  --optimizer sgd \
  --lr 0.002 \
  --weight-decay 5e-4 \
  --momentum 0.9 \
  --scheduler multistep \
  --lr-milestones 40 70 \
  --lr-gamma 0.1 \
  --grad-clip-norm 10 \
  --selection-metric f1_macro \
  --early-stopping-patience 10 \
  --num-workers 4 \
  --device cuda \
  --output-dir /root/project/paper_attention_yolov3_repro/outputs_paper_detector_protocol_a_lr2e3_autodl
```

## 9. Quick checks

Check GPU use:

```bash
nvidia-smi
```

Check whether training is still running:

```bash
ps -ef | grep train_detector.py
ps -ef | grep train_classifier.py
tail -f train.log
```

## 10. Notes

- In this repo, training scripts default to `--device cpu`, so on AutoDL you should explicitly pass `--device cuda`.
- Keep large datasets outside Windows-style paths and preferably under short ASCII-only Linux paths.
- If the instance is powered off, data is kept, but do not leave it powered off for too long without checking AutoDL retention rules.
