# U-net paper Reproduction Report
![alt text](/figs/teaserimage.png "U-net reproduction")

by: 
* Vera Hoveling, V.T.Hoveling@student.tudelft.nl, 4591941
* Sayra Ranjha S.S.Ranjha@student.tudelft.nl, 4xxxxx
* Maaike Visser, M.E.B.P.Visser@student.tudelft.nl, 4597265

## Introduction
This document details our reproduction of the now seminal paper "U-net: Convolutional Networks for Biomedical Image Segmentation", by Ronneberger et al (https://arxiv.org/abs/1505.04597). This project was undertaken as part of the Deep Learning course (CS4240) at the Delft University of Technology. 
![alt text](/figs/u-net-architecture.png "U-net architecture")*U-Net architecture*

In this document we briefly explain the U-Net architecture, as well as the steps we took to reproduce the results 


## U-Net 
[small explanation of U-Net]

 In this notebook we briefly go over the original ResNet paper https://arxiv.org/abs/1512.03385 and explain the steps to reproduce the Table 6 in the paper.

The Pytorch implementation of the ResNet architecture used in the notebook is based on the publicly available code in

https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
https://github.com/akamaster/pytorch_resnet_cifar10.
We changed the code so that training and testing can be performed in Jupyter notebook via Google Colab.

The results from the original paper have been successfully reproduced with better or comparable results except for the ResNet1202 model since Google Colab does not support enough (16GB) memory usage on GPU for this experiment. We also provided the pretrained models in this repository for evaluation purposes.

## Implementation
We have implemented the network both with Pytorch and Keras. The same hyperparameter settings were used for both implementations, showing not only that it is possible to reproduce the work but also inspect differences one might encounter when using another framework.

## Training
Training with Keras

![alt text](/figs/learning_curve_opt_SGD__100eps_wvalidation_lrscheduling.png "Training in Keras")

# Results

| framework | intersection over union | pixel error         |   |   |
|-----------|-------------------------|---------------------|---|---|
| keras     | 0.880027184589537       | 0.09784660339355469 |   |   |
| pytorch   |                         |                     |   |   |

## Motivation of choices / Discussion

### Data augmentation
### Confusion stuff on upsampling/transposed convolution/upconvolution
### All other questions for authors

## Future work
