# HR-MFNet

Hierarchical Refinement Network with Multi-Frequency Interaction for retinal vessel segmentation.

## 1. Overview

This repository provides HR-MFNet training and inference code, with a unified entry point at `main.py`.

- Training config: `configs/train.yml`
- Inference config: `configs/inference.yml`
- Training script: `bash_train.sh`
- Inference script: `bash_inference.sh`

## 2. Environment

Recommended setup:

- Linux (Ubuntu 22.04+)
- Python 3.9+
- CUDA12.8 + PyTorch2.5.0


## 3. Dataset Layout

Recommended directory structure:

```text
data/
	DATASET_NAME/
		train/
			input/
			label/
			fov/
		val/
			input/
			label/
			fov/
```

Example for DRIVE:

```text
data/DRIVE/train/input
data/DRIVE/train/label
data/DRIVE/train/fov
data/DRIVE/val/input
data/DRIVE/val/label
data/DRIVE/val/fov
```

## 4. Supported Dataset Names

Use one of the following values for `dataset_name`:

- `DRIVE`
- `CHASE_DB1`
- `OCTA500_3MM`
- `OCTA500_6MM`
- `DAC1`


When the following fields are set to `auto`, they are filled automatically based on `dataset_name`:

- `train_x_path / train_y_path / train_z_path`
- `val_x_path / val_y_path / val_z_path`
- `input_size`
- `transform_rand_crop`

## 5. Training

1. Edit `configs/train.yml` and verify at least:

- `mode: train`
- `dataset_name`
- `data_root`
- `model_name: 'HR_MFNet'`
- `CUDA_VISIBLE_DEVICES`
- `wandb`

2. Start training:

```bash
bash bash_train.sh
```

Equivalent command:

```bash
python3 main.py --config_path "configs/train.yml"
```

Checkpoints are saved by default under `model_ckpt/<timestamp>/`.

## 6. Inference

1. Edit `configs/inference.yml` and verify at least:

- `mode: inference`
- `dataset_name`
- `data_root`
- `model_name: 'HR_MFNet'`
- `model_path` (must point to an existing `.pt` file)
- `CUDA_VISIBLE_DEVICES`

2. Start inference:

```bash
bash bash_inference.sh
```

Equivalent command:

```bash
python3 main.py --config_path "configs/inference.yml"
```

Inference outputs are saved to:

```text
<directory of model_path>/<model filename without extension>/
```

Each sample exports `_argmax.png` and `_target.png`.


## 7. Reported Results

| Dataset    | ACC   | SE    | SP    | F1    | MIoU  | AUC   |
| ---------- | ----- | ----- | ----- | ----- | ----- | ----- |
| DRIVE      | 97.33 | 83.31 | 98.54 | 82.90 | 70.85 | 97.96 |
| CHASE_DB1  | 97.73 | 82.92 | 98.38 | 81.50 | 86.61 | 99.47 |
| OCTA500-6M | 97.90 | 88.37 | 98.89 | 88.84 | 79.91 | 98.61 |
| OCTA500-3M | 98.89 | 91.25 | 99.44 | 91.54 | 84.63 | 98.67 |
| DAC1       | 97.70 | 83.64 | 98.50 | 79.07 | 65.73 | 98.46 |

## 8. Pretrained Weights

Release: https://github.com/nliang1995/HR-MFNet/releases/tag/weight_20260318

## 12. Citation


```