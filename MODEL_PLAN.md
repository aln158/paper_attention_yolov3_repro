# Reproduction plan

## Paper target

Paper: `An Attention Deep Learning Framework-Based Drowsiness Detection Model for Intelligent Transportation System`

The paper combines three ideas in one pipeline:

1. A `Darknet-53` style feature extractor.
2. A spatial attention branch with a learnable scalar `gamma`.
3. A `YOLOv3` style three-scale detection head plus a global-average-pooling classification head used for drowsiness prediction and CAM visualization.

## Backbone layout

Input size in the paper: `416 x 416 x 3`

Backbone stages reproduced here:

1. `Conv(3x3, 32)` -> `Conv(3x3, 64)`
2. `MaxPool`
3. `1 x ResidualBlock(64, hidden=32)`
4. `Conv(3x3, 128)` -> `MaxPool`
5. `2 x ResidualBlock(128, hidden=64)`
6. `Conv(3x3, 256)` -> `MaxPool`
7. `8 x ResidualBlock(256, hidden=128)` -> high-resolution feature map
8. `Conv(3x3, 512)` -> `MaxPool`
9. `8 x ResidualBlock(512, hidden=256)` -> mid-resolution feature map
10. `Conv(3x3, 1024)` -> `MaxPool`
11. `4 x ResidualBlock(1024, hidden=512)` -> deep feature map

## Attention branch

The paper describes a spatial attention map `H(F(X))` and a learnable weight `gamma`.

In this reproduction:

1. The deepest `1024`-channel feature map is projected to a single-channel attention score map.
2. A spatial softmax is applied over the full `H x W` grid.
3. The attention weights modulate the deep feature map.
4. The final attention-enhanced feature is:

`attended_feature = feature + gamma * (feature * softmax(score(feature)))`

## Detection neck and heads

The reproduction keeps a three-scale `YOLOv3` style neck:

1. Deep feature -> detection head (`13 x 13` when input is `416`)
2. Upsample and fuse with mid feature -> detection head (`26 x 26`)
3. Upsample and fuse with high feature -> detection head (`52 x 52`)

Each detection head predicts:

`anchors_per_scale x (4 bbox + 1 objectness + num_detection_classes)`

## Classification head

The paper also uses global average pooling and CAM-based interpretation. The reproduction does the same:

1. Attention-enhanced deep feature
2. `AdaptiveAvgPool2d(1)`
3. `Dropout`
4. `Linear(1024 -> num_classes)`
5. CAM from classifier weights and the deep feature map

## Local data adaptation

The local workspace currently provides a classification-style image dataset under:

`prepared_utarldd_mc_real/train`

That dataset does not include YOLO bounding-box annotations. Because of that, the reproduction is split into two practical tracks:

1. `train_classifier.py`
   Trains the paper-style attention backbone and classification head directly on the local images.
2. `train_detector.py`
   Trains the full detection branch when a YOLO-format dataset is available.

## What is reproduced exactly vs approximately

Reproduced closely:

- Darknet-53 inspired backbone depth
- Spatial attention with learnable `gamma`
- Three-scale YOLO-style detector heads
- Global-average-pooling classification head
- CAM generation path
- Adam-based training entry points

Approximated:

- The paper's exact Keras implementation details
- The exact anchor values, because the paper does not provide them explicitly
- Joint detection training on the local dataset, because local labels are classification-only

## Validation strategy in this workspace

1. Build the full PyTorch model.
2. Run a tensor-shape smoke test.
3. Run the classification path on the local UTARLDD-style images.
4. Save a checkpoint and a CAM visualization example.
