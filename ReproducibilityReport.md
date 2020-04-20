# U-net paper Reproduction Report
![alt text](/figs/teaserimage.png "U-net reproduction")

by: 
* Vera Hoveling, V.T.Hoveling@student.tudelft.nl, 4591941
* Sayra Ranjha S.S.Ranjha@student.tudelft.nl, 4555449
* Maaike Visser, M.E.B.P.Visser@student.tudelft.nl, 4597265

## Introduction
This document details our reproduction of the now seminal paper "U-net: Convolutional Networks for Biomedical Image Segmentation", by Ronneberger et al (https://arxiv.org/abs/1505.04597). This project was undertaken as part of the Deep Learning course (CS4240) at the Delft University of Technology. 
![alt text](/figs/u-net-architecture.png "U-net architecture")*U-Net architecture*

In this document we briefly explain the U-Net architecture, as well as the steps we took to reproduce the "Pixel Error" the result from Table 1. We also include a quality measure not present in the original paper for the specific data set we used, namely the Intersection over Union. 

Pre-trained models are also available for evaluation purposes. 


## U-Net 
[small explanation of U-Net]

 
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
