# DiverseImageCaptioning
This is an implementation of the paper [**On diversity in image captioning: metrics and methods**](https://doi.ieeecomputersociety.org/10.1109/TPAMI.2020.3013834) and [**Towards Diverse and Accurate Image Captions via Reinforcing Determinantal Point Process**](https://arxiv.org/abs/1908.04919).

## Requirements
Python 2.7

Pytorch 0.4

java 1.8

## Data preparation and pre-training
Please refer to README-DISC.md to train the Att2in model. Note that you can just train the model using cross-entropy for 30 epochs.

## Training using diversity reward
XE denotes cross-entropy loss, CIDEr denotes CIDEr reward, DISC denotes retrieval rewards proposed in [**DISCCap**](https://openaccess.thecvf.com/content_cvpr_2018/papers/Luo_Discriminability_Objective_for_CVPR_2018_paper.pdf), DIV denotes [**Self-CIDEr**](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Describing_Like_Humans_On_Diversity_in_Image_Captioning_CVPR_2019_paper.html) and DPP denotes determinantal point process rewards, which combines the quality (CIDEr) and diversity (self-CIDEr) into one matrix and the determinant of the matrix reflects both diversity and quality.
### Training using XE+CIDEr+DISC
```
xe=1
cider=10
disc=1
div=0
naiverl=0
numc=1
bash train_diversity.sh $xe $cider $disc $div selfcider $naiverl $numc
```
The above values denote the weight of the loss functions and ```selfcider``` denote the diversity function, you can also set it to ```LSA``` or ```mcider```. For ```naiverl```, set it to 0, works better and ```numc``` represents the number of captions that should be drawn from ```p(c|I)```, and in this section 1 is fine.
### Training using CIDEr+DIV
```
xe=0
cider=1
disc=0
div=1
naiverl=0
numc=5
bash train_diversity.sh $xe $cider $disc $div selfcider $naiverl $numc
```
In this section, to compute the diversity reward, you need to set ```numc``` to larger than 1.

### Taining using DPP
```
numc=5
subset=-1
retrieval=1
bash train_dpp.sh $numc $subset $retrieval
```
```numc``` must be larger than 1 and ```subset``` should be -1. If you want to use retrieval reward as the quality function, please set ```retrieval``` to a value that larger than 0, otherwise, only CIDEr is used as the quality function.

## Inference
```bash eval.sh $model_id test random_sample $num_samples```
```num_samples``` denotes the number of captions you want to generate for each image.

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
