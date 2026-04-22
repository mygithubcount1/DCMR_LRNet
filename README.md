# Dual-Branch Cross-Layer Feature Fusion and Multi-Region Prediction Ensemble for Ultra-Fine-Grained Plant Leaf Recognition

This repository contains the official implementation of the paper:

> *Dual-Branch Cross-Layer Feature Fusion and Multi-Region Prediction Ensemble for Ultra-Fine-Grained Plant Leaf Recognition**
> Submitted to *The Visual Computer* (2026).

## Requirement

- Python 3.8.20
- torch 1.13.0
- timm 1.0.15
- numpy 1.21.6
- pandas 1.1.5
- Pillow  9.3.0

## Dataset Availability

You can download the datasets from the links below:

+ [SoyLocal, SoyGlobal, SoyGene, SoyAge, and Cotton](https://pan.baidu.com/s/1bPJYmFGWJg2eTr5Ipfw6uA). Access code: iccv
+ [SoyCultivar200](https://drive.google.com/file/d/1XsWZPYYrDsCwAy5r4t3I1F_lOOrGGhgf/view)

## Run the experiments

**Please run `texture_extractor_images4.py` to extract texture images before conducting experiments.**

Run train.py to train the model, e.g., train on the Cotton dataset with EfficientNet-B0.

    $ python train.py  --dataset COTTON --save_model True --save_dirname weights_cotton

Run train.py to train the model, e.g., train on the Cotton dataset with ResNet-50.

    $ python train_resnet.py  --dataset COTTON --save_model True --save_dirname weights_resnet_cotton

 ## 🔗 Related Information

This repository is directly associated with the manuscript submitted to The Visual Computer.