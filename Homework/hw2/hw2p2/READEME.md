# Face Classification and Verification

## About

> This is a course project from CMU 11-785 Introduction to deep learning. 

Given an image of a person's face, **face classification** stands for classifying its ID, and **face verification** stands for determining whether two face images are of the same person. 

In this project, Convolutional Neural Networks is used to build an end-to-end system for both tasks. For the face classification task, the system takes an image of a face as input and outputs the ID of the face. For the verification task, the system takes a pair of images as input and outputs a score that quantifies the similarity the two faces, or whether the two faces belong to the same person.

## Table of Contents
* [Overview](#overview)
* [Key Learnings](#key-learnings)
* [Installation](#installation)
* [Usage](#usage)

## Overview

The project is to be trained on a dataset with a few thousand images with labelled ID's. For the classification task, the input to the system is a face image and it will predict the ID of the face. The ground truths are present in the training data ,so the network will be doing an N-way classification to get the prediction. For the verification task, the input to the system is a pair of face images. The goal is to output a numeric score that quantifies how similar the faces in the two images are. A higher score will indicate higher confidence that the faces in the two images are of the same person.

### Approach

I implement a ResNet model, supporting 50-layer, 101-layer and 152-layer configurations for both tasks. The model is trained for classification task so that the network did an N-classification to get the predication. Suppose the labeled dataset contains a total of M images that belong to N different people. The output of the last such feature extraction layer (before the last fully connected layer) is the *face embedding*. The resulting embeddings will encode a lot of discriminative facial features.

After training, given two images, each image will be passed through the network to generate face embeddings. Specially, the face embeddings are fix-length vectors. I use *cosine similarity* score (cosine value of the two vectors) to produce the similarity score. A threshold is needed so that scores higher than it indicates the two persons are of the same person. For a given threshold, four conditions can happen: *false positive*, *false negative*, *true negative* and *true positive*. The **Receiver Operating Characteristic (ROC)** created by plotting the **True Positive Rate (TPR)** against the **False Positive Rate (FPR)** at various threshold settings. The Area Under the Curve (AUC) for the ROC curve is equal to the probability that a classifier will rank a randomly chosen similar pair (images
of same people) higher than a randomly chosen dissimilar one (images from two different people).



## Key Learnings

* I develope skills in building Convolution Neural Networks as effective shift-invariant feature extractors, and tune model and network parameters for a specific task.
* I improve the training performance with two auxilary losses. **Center loss** calculates the centriod of feature vectors of the same class and penalizes those distant from the center. It helps features of the same class to get closer, but cannot contributes to seperate features of different classes. Also, a scale factor is needed to be tuned to balance this loss with the cross-entropy loss. **Angle loss** expresses the decision boundray as angles in a hyper-spherical space. It adds weights to the angles of each class, so that the desision boundary is forced to seperate different classes as far as possible. However, I find these losses are aggresive, and can only be used to boost performance when the model converges. If they are used at the beginning of the training, the model will diverge. 
* I get famaliar with the evaluation metrics for face verifcation tasks
* I develop skills for processing and training neural networks with big data.

## Installation

Install [Pytorch](https://pytorch.org/). The code has been tested with Python 3.7, Pytorch 1.4, CUDA 10.1 on Ubuntu 18.04.

* Use pip to install required Python packages

    ```
    pip install -r requirements
    ```

* Dataset can be downloaded from [Dataset](https://drive.google.com/file/d/1-K2YgajOCNtggFKVr4Zj05cFUhDaZsRJ/view?usp=sharing)

## Usage

The training data has two parts (*medium* and *large*). The medium dataset is used for setting up the model and pretraining, while the large dataset is for finetuning the parameters of the model. Auxilary losses (only support *AngleLoss* and *CenterLoss* now) can be optionally added when the training stagnates, but are not recommended to applied from the beginning.

* Start training from scratch by
    ```
    python main.py --data_path=YOUR_DATASET_PATH --mode=pretrain --loss=None
    ```

* Evaluate with the pretrained [model]() by
    ```
    python main.py --data_path=YOUR_DATASET_PATH --mode=eval --loss=None
    ```

* Continue training using center loss on medium dataset with the [model]() by 
    ```
    python main.py --data_path=YOUR_DATASET_PATH --mode=cont_pre --loss=center
    ```

* Continue training using center loss on large dataset with either of the prev two models by
    ```
    python main.py --data_path=YOUR_DATASET_PATH --mode=fine_tune --loss=center
    ```

The models are trained with a machine of two GPUs. If you don't have two or more GPUs, please train from scratch. The model converges in about 50~60 epochs.





