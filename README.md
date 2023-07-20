<p align="center">
<img src=./img/BRG.png width=40% height=40%>
</p>

# Point Transformer based Auto-Encoder for Robot Grasping and Quality Inference

This code is the implementation of a project that I have worked on for the [Barton Research Group](https://brg.engin.umich.edu/research/robotic-smart-manufacturing).
In this project we studied how transformer blocks that are set operators are particularly well suited for point clouds and thus make auto-encoders very efficient on point clouds, conserving their permutation invariance and study the application of such algorithms on an application for Robot Grasping.

<p align="center">
<img src=./img/pres.png width=40% height=40%>
</p>

<p align="center">
<img src=./img/overview.png width=40% height=40%>
</p>

# Contents

[***Objective***](https://github.com/leob03/PTAE#objective)

[***Concepts***](https://github.com/leob03/PTAE#concepts)

[***Overview***](https://github.com/leob03/PTAE#overview)

[***Dependencies***](https://github.com/leob03/PTAE#dependencies)

[***Getting started***](https://github.com/leob03/PTAE#getting-started)

[***Deeper dive into the code***](https://github.com/leob03/PTAE#deeper-dive-into-the-code)

# Objective

**To encode the information of the surfaces from a point cloud representation of a CAD file into a more meaningful latent space useful to determine grasping point candidates based on physical parameters.**


# Concepts

* **Encoder-Decoder architecture**. Typically, a model that generates sequences will use an Encoder to encode the input into a fixed form and a Decoder to decode it, word by word, into a sequence.

* **Attention**. The use of Attention networks is widespread in deep learning, and with good reason. This is a way for a model to choose only those parts of the encoding that it thinks is relevant to the task at hand. The same mechanism you see employed here can be used in any model where the Encoder's output has multiple points in space or time. In image captioning, you consider some pixels more important than others. In sequence to sequence tasks like machine translation, you consider some words more important than others.

* **Transfer Learning**. This is when you borrow from an existing model by using parts of it in a new model. This is almost always better than training a new model from scratch (i.e., knowing nothing). As you will see, you can always fine-tune this second-hand knowledge to the specific task at hand. Using pretrained word embeddings is a dumb but valid example. For our image captioning problem, we will use a pretrained Encoder, and then fine-tune it as needed.

# Overview

The pipeline for the project looks as follows:

- The **input** is a dataset of images and 5 sentence descriptions that were collected with Amazon Mechanical Turk. We will use the 2014 release of the [COCO Captions dataset](http://cocodataset.org/) which has become the standard testbed for image captioning. The dataset consists of 80,000 training images and 40,000 validation images, each annotated with 5 captions.
- In the **training stage**, the images are fed as input to RNN (or LSTM/LSTM with attention depending on the model) and the RNN is asked to predict the words of the sentence, conditioned on the current word and previous context as mediated by the hidden layers of the neural network. In this stage, the parameters of the networks are trained with backpropagation.
- In the **prediction stage**, a witheld set of images is passed to RNN and the RNN generates the sentence one word at a time. The code also includes utilities for visualizing the results.

# Dependencies
**Python 3.10**, modern version of **PyTorch**, **numpy** and **scipy** module. Most of these are okay to install with **pip**. To install all dependencies at once, run the command `pip install -r requirements.txt`

I only tested this code with Ubuntu 20.04, but I tried to make it as generic as possible (e.g. use of **os** module for file system interactions etc. So it might work on Windows and Mac relatively easily.)


# Getting started

1. **Get the code.** `$ git clone` the repo and install the Python dependencies
2. **Train the models.** Run the training `$ python train_rnn.py` or `$ python train_lstm.py` or `$ python train_lstm_attention.py`, depending on the model that you want to try (see many additional argument settings inside the file) and wait. You'll see that the learning code writes checkpoints into `cv/` and periodically print its status. 
3. **Evaluate the models checkpoints and Visualize the predictions.** To evaluate a checkpoint from `checkpoints/`, run the scripts `$ python test_rnn.py` or `$ python test_lstm.py` or `$ python test_lstm_attention.py` and pass it the path to a checkpoint ( by adding --checkpoint /path/to/the/checkpoint after your python command).

# Deeper dive into the code


