<p align="center">
<img src=./img/BRG.png width=40% height=40%>
</p>

# Point Transformer based Auto-Encoder for Robot Grasping and Quality Inference

This code is the implementation of a project that I have worked on for the [Barton Research Group](https://brg.engin.umich.edu/research/robotic-smart-manufacturing).
In this project we studied how Transformer Blocks that are set operators are particularly well suited for Point-Clouds and thus can make Auto-Encoders very efficient on point clouds, conserving their permutation invariance. We also worked on a concrete application of our method for Robot Grasping.

&nbsp;

<p align="center">
  <img src=./img/pres.png width="700" height="300">
  <br>
  Overview of the Auto-Encoder
</p>

<p align="center">
  <img src=./img/overview.png width="700" height="250">
  <br>
  Overview of where it intervenes in the whole process
</p>

# Contents

[***Objective***](https://github.com/leob03/PTAE#objective)

[***Concepts***](https://github.com/leob03/PTAE#concepts)

[***Overview***](https://github.com/leob03/PTAE#overview)

[***Dependencies***](https://github.com/leob03/PTAE#dependencies)

[***Getting started***](https://github.com/leob03/PTAE#getting-started)

[***Notes***](https://github.com/leob03/PTAE#notes)

# Objective

**To encode the information of the surfaces from a point cloud representation of a CAD file into a more meaningful latent space useful to determine grasping point candidates based on physical parameters.**


# Concepts

Most of the concepts are described quite in depth in the paper (reference the paper) but here is a quick summary of the main concepts exploited in this project:

* **Encoder-Decoder architecture**. Typically, a model that generates sequences will use an Encoder to encode the input into a fixed form and a Decoder to decode it, word by word, into a sequence.

* **Attention**. The use of Attention networks is widespread in deep learning, and with good reason. This is a way for a model to choose only those parts of the encoding that it thinks is relevant to the task at hand. The same mechanism you see employed here can be used in any model where the Encoder's output has multiple points in space or time. In image captioning, you consider some pixels more important than others. In sequence to sequence tasks like machine translation, you consider some words more important than others.

* **Transfer Learning**. This is when you borrow from an existing model by using parts of it in a new model. This is almost always better than training a new model from scratch (i.e., knowing nothing). As you will see, you can always fine-tune this second-hand knowledge to the specific task at hand. Using pretrained word embeddings is a dumb but valid example. For our image captioning problem, we will use a pretrained Encoder, and then fine-tune it as needed.

# Overview

The pipeline for the project looks as follows:

- The **input** is a dataset of images and 5 sentence descriptions that were collected with Amazon Mechanical Turk. We will use the 2014 release of the [COCO Captions dataset](http://cocodataset.org/) which has become the standard testbed for image captioning. The dataset consists of 80,000 training images and 40,000 validation images, each annotated with 5 captions.
- In the **training stage**, the images are fed as input to RNN (or LSTM/LSTM with attention depending on the model) and the RNN is asked to predict the words of the sentence, conditioned on the current word and previous context as mediated by the hidden layers of the neural network. In this stage, the parameters of the networks are trained with backpropagation.
- In the **prediction stage**, a witheld set of images is passed to RNN and the RNN generates the sentence one word at a time. The code also includes utilities for visualizing the results.

# Dependencies
**Python 3.10**, modern version of **PyTorch**, **numpy** and **scipy** module. Most of these are okay to install with **pip**. To install the rest of the dependencies all at once, run the command `./install.sh`

I only tested this code with Ubuntu 20.04, but I tried to make it as generic as possible (e.g. use of **os** module for file system interactions etc. So it might work on Windows and Mac relatively easily.)


# Getting started

1. **Get the code.** `$ git clone` the repo and install the Python dependencies.
2. To run a predefined simple demo of the code and test the main results run the command `python demo.py`

<p align="center">
  <img src=./img/archAE.png width="700" height="300">
  <br>
  The Neural Network architecture of our Auto-Encoder
</p>

## Testing the Encoder on 3D Shape Classification task:

To train our encoder on 3D Shape Classification task we had to make some small modifications on the network architecture in order to interpret and reduce the latent space into a set of vectors representing the object category labels. To this extent we added a Global Average Pooling Layer before the final projection layer to which we also adapted the dimensions in order to get a vector of the size of the number of categories. Fig 6 represents the structure used for 3D Shape Classification.

Then, we evaluated it on ModelNet40, a dataset containing 12,311 CAD models with 40 object categories. They are split into 9,843 models for training and 2,468 for testing. As in PointNet++ [3] we preprocessed the data through uniform sampling of each CAD model together with the normal vectors from the object meshes. For evaluation metrics, we use the mean accuracy within each category and the overall accuracy over all classes. (The results are in the paper)

<p align="center">
  <img src=./img/3DShapeClass.png width="700" height="300">
  <br>
  The modified architecture of our encoder to test it on 3D Shape Classification
</p>

1. **Get the data.** Download alignment **ModelNet** [here](https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip) and save in `modelnet40_normal_resampled`.
2. **To train the model for Classification:** Run the training `$ python train/train_classification.py`. You'll see that the learning code writes checkpoints into `log/cls/Leo/train_classification.log` and periodically print its status. 
3. **Evaluate the models checkpoints.** To evaluate a checkpoint from `models/`, run the scripts `$ python eval_classification.py`. You'll see that the learning code writes checkpoints into `log/cls/Leo/eval_classification.log` and periodically print its status. 

## Testing the Auto-Encoder on 3D Part-Segmentation task:

To train our whole network on 3D Shape Classification task we also had to make some small modifications on the architecture. Indeed, an auto-encoder is only expected to output a reconstruction of the input in its initial structure with a little degradation due to the loss of information in the data compression for the latent space. However, to perform part- segmentation we had to assign each point to a part label.

Then, we evaluated it on the ShapeNetPart dataset which is annotated for 3D object part segmentation. It consists of 16,880 models from 16 shape categories, with 14,006 3D models for training and 2,874 for testing. The number of parts for each category is between 2 and 6, with 50 different parts in total. In our case we only kept 4 parts: the airplane which as 4 parts, the bag which as 2 parts, the cap which as 2 parts and the car which as 4 parts. Again we preprocessed the data using PointNet++[3] sampling technique. For evaluation metrics, we report category mIoU and instance mIoU. (The results are in the paper)

<p align="center">
  <img src=./img/3DPartSeg.png width="700" height="300">
  <br>
  The architecture of our auto-encoder to test it on 3D Part Segmentation
</p>

1. **Get the data.** Download alignment **ShapeNet** [here](https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip) and save in `data/shapenetcore_partanno_segmentation_benchmark_v0_normal`.
2. **To train the model for Part Segmentation:** Run the training `$ python train/train_partsegmentation.py`. You'll see that the learning code writes checkpoints into `log/partseg/Leo/train_classification.log` and periodically print its status. 
3. **Evaluate the models checkpoints.** To evaluate a checkpoint from `models/`, run the scripts `$ python eval_partsegmentation.py`. You'll see that the learning code writes checkpoints into `log/partseg/Leo/eval_partsegmentation.log` and periodically print its status. 

# Notes
Some code and training settings are borrowed from both https://github.com/yanx27/Pointnet_Pointnet2_pytorch and from the paper [PCT: Point Cloud Transformer (Meng-Hao Guo et al.)](https://arxiv.org/abs/2012.09688) which original implementation is accessible here: https://github.com/MenghaoGuo/PCT.

