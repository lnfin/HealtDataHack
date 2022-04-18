# Health Data Hack

## Introduction
This is our team’s 4th place solution for [Healt Data Hack](https://codenrock.com/contests/hackhealth#/) (second task). This contest was about segmentation of colorectal cancer cells on high resolution histological slices. Competition data was prepared by MIPT University, medtech.moscow and Phystech School of Biological and Medical Physics.

## Solution
### Splitting images
We did a [cutter.py](modeling/cutter.py) module that pads each side of the image to %patch_size length and then goes with sliding windows (window size equal to patch_size). In training we used two different patch sizes: 1024x1024 and 2048x204 with 50% overlap:

<img width="1103" alt="2022-04-18_15-29" src="https://user-images.githubusercontent.com/54595287/163815606-359b53ea-e8da-48a8-af5d-370a845d1559.png">  

### Tresholding
The dataset was segregated by thresholding it for a minimum X% of tissue pixels. After thresholding for several percentages such as 40%, 30%, 20%, etc., it was observed that by thresholding with 30%, maximum redundancy was removed, and useful information was saved.

### TTA
For TTA we used simple averaging of the default image and augmentated:
- Horizontally flipped
- Vertically flipped
- Rotated (90, 180, 270 angles)

### Final ensemble
Our final solution contains [two models](https://drive.google.com/drive/folders/1561kJfurS61cxtOjkpOh0-6pmhNPXPiW?usp=sharing):
1. Unet++ with EffNetb7 backbone and 2048x2048 patch size.
2. Unet++ with EffNetb7 backbone and 1024x1024 patch size.

We tried different types of Ensembles (MaxProb, MinProb, MeanProb) and Simple Averaging Ensemble obtain the best score.

## Project structure
- **FirstLook.ipynb** - exploratory data analysis notebook
- **Training.ipynb** - notebook for training all models
- **Inference.ipynb** - inference notebook
- **productions.py** - preparing test data for prediction
- **train_functions.py** - module for training and validation
- **modeling**
  - cutter.py - module for splitting images
  - losses.py - custom loss functions
  - metrics.py - custom metrics
  - models.py - custom models
- **utils**
  - cfgtools.py - configuration file
  - datagenerator.py - module for preparing data for training
  - dataset.py - module for preparing data for training

_Additional data_:
[train data](https://drive.google.com/file/d/1erA0TiUZb2os-QJ-vFN_K1bhNBroBLpO/view),
[test data](https://drive.google.com/file/d/1NUQjp10RmymKohg0cjqL6G3fRk_DpVsH/view),
[weights](https://drive.google.com/drive/folders/1561kJfurS61cxtOjkpOh0-6pmhNPXPiW?usp=sharing), 
[configs](https://drive.google.com/drive/folders/1frbD1cqIEN_fpoyd6GRSKYKS2sNpgVMU?usp=sharing),
[presentation](https://docs.google.com/presentation/d/1SN-Olu-dxH2VZAZuMpiBncV508JN0pVh/edit?usp=sharing&ouid=116202268270672729224&rtpof=true&sd=true), 
[text](https://docs.google.com/document/d/1RKJ9ijLmTFxTVfYAuHraC5Q5J9zVKxcr/edit?usp=sharing&ouid=116202268270672729224&rtpof=true&sd=true)
