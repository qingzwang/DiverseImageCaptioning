# DiverseImageCaptioning
This is an implementation of the paper [**On diversity in image captioning: metrics and methods**](https://doi.ieeecomputersociety.org/10.1109/TPAMI.2020.3013834) and [**Towards Diverse and Accurate Image Captions via Reinforcing Determinantal Point Process**](https://arxiv.org/abs/1908.04919).

## Requirements
Python 2.7

Pytorch 0.4

java 1.8

## Data preparation and pre-training
Please refer to README-DISC.md to train the Att2in model. Note that you can just train the model using cross-entropy for 30 epochs.

## Training using diversity reward
### Training using XE+CIDEr
### Training using CIDEr+DIV
### Taining using DPP

## Ackonwledgement
This repository is based on [**DISCCap**](https://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Discriminability_Objective_for_CVPR_2018_paper.pdf) and we would like to thank the author [**Ruotian Luo**](https://ttic.uchicago.edu/~rluo/).

## Citation
If you think this repository is helpful please cite the following papers:
```
@article{wang2020tpami,
author={Qingzhong Wang and Jia Wan and Antoni B. Chan},
title={On Diversity in Image Captioning: Metrics and Methods},
journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
year={2020}
}

@article{wang2019towards,
  title={Towards Diverse and Accurate Image Captions via Reinforcing Determinantal Point Process},
  author={Wang, Qingzhong and Chan, Antoni B},
  journal={arXiv preprint arXiv:1908.04919},
  year={2019}
}
```
