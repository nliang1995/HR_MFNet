# Full-scale Representation Guided Network for Retinal Vessel Segmentation
Official repository of the paper HR-MFNet: Hierarchical Refinement Network with Multi-Frequency Interaction
for Retinal Vascular Structure Segmentation



## Environment

- OS: Ubuntu 22.04 LTS
- GPU: RTX 4090 24GB
- CUDA: 12.4
- Pytorch 2.5.0

## ✅ Experimental Result

|Dataset|ACC|SE|SP|F1|MIoU|AUC|
|---|---|---|---|---|---|---|
|DRIVE|97.33|83.31|98.54|82.90|70.85|97.96|
|CHASE_DB1|97.73|82.92|98.38|81.50|86.608|99.47|
|OCTA500-6M|97.90|88.37|98.89|88.84|79.91|98.61|
|OCTA500-3M|98.89|91.25|99.44|91.54|84.63|98.67|
|DAC1|97.70|83.64|98.50|79.07|65.73|98.46|


## ✅ Pretrained model for each dataset
Each pre-trained model could be found on [release version](https://github.com/ZombaSY/FSG-Net-pytorch/releases/tag/1.1.0)


## 🧻 Dataset Preparation
You can edit `train_x_path...` in [<b>configs/train.yml</b>](configs/train.yml) <br>
The input and label should be sorted by name, or the dataset is unmatched to learn.

For train/validation set, you can download from public link or [release version](https://github.com/ZombaSY/FSG-Net-pytorch/releases/tag/1.1.0)

---

## 🚄 Train

If you have installed 'WandB', login your ID in command line.<br>
If not, modify `wandb=false` in [<b>configs/train.yml</b>](configs/train.yml).<br>
You can login through your command line or `wandb.login()` inside "main.py"

For <b>Train</b>, edit the [<b>configs/train.yml</b>](configs/train.yml) and execute below command
```
bash bash_train.sh
```

---

## 🛴 Inference

For <b>Inference</b>, edit the [<b>configs/inference.yml</b>](configs/inference.yml) and execute below command. <br>
Please locate your model path via `model_path` in [<b>configs/inference.yml</b>](configs/inference.yml)</b>
```
bash bash_inference.sh
```

- If you are using pretrained model, the result should be approximate to experimental result's

## 📚 Citation
```bibtex

```
