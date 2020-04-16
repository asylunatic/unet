# U-net paper Reproduction Report
![](/figs/teaserimage.png "U-net reproduction")
by: 
* Vera Hoveling, V.T.Hoveling@student.tudelft.nl, 4591941
* Sayra Ranjha S.S.Ranjha@student.tudelft.nl, 4xxxxx
* Maaike Visser, M.E.B.P.Visser@student.tudelft.nl, 4597265

# Intro
This document details our reproduction of the now seminal paper U-net: Convolutional Networks for Biomedical Image Segmentation, by Ronneberger et al (https://arxiv.org/abs/1505.04597). 

# Implementation
We have implemented the network both with Pytorch and Keras. The same hyperparameter settings were used for both implementations, showing not only that it is possible to reproduce the work but also inspect differences one might encounter when using another framework.

# Training


# Results

| framework | intersection over union | pixel error         |   |   |
|-----------|-------------------------|---------------------|---|---|
| keras     | 0.880027184589537       | 0.09784660339355469 |   |   |
| pytorch   |                         |                     |   |   |

# Motivation of choices / Discussion

## Data augmentation
## Confusion stuff on upsampling/transposed convolution/upconvolution
## All other questions for authors

# Future work
