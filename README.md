# GAN-Project
GAN (Generative Adversarial Network) 
Overview
This repository contains the implementation of a Generative Adversarial Network (GAN) for generating fashion images using the Fashion MNIST dataset. The GAN architecture includes a generator and a discriminator trained on TensorFlow.

Project Structure
gan.py: Defines the GAN model, generator, discriminator, and custom training loop.
model_monitor.py: Custom callback for monitoring and saving generated images during training.
train_gan.py: Script for training the GAN using the Fashion MNIST dataset.
images/: Directory to store generated images during training.
generator_model.h5: Saved generator model.
gan_weights.h5: Saved GAN weights.
Prerequisites
Make sure to install the required packages using:

Dataset
The GAN is trained on the Fashion MNIST dataset, loaded and preprocessed using TensorFlow Datasets (tfds). The images are resized to 28x28 pixels and normalized.

Model Architecture
Generator: Sequential model with dense and upsampling layers.
Discriminator: Sequential model with convolutional layers.
Training
To train the GAN, run the following command:

The training script will save generated images in the images/ directory and store the generator model and GAN weights.

Results
Generated images are saved in the images/ directory during training. The final generator model is saved as generator_model.h5, and GAN weights are saved as gan_weights.h5.

References
TensorFlow: https://www.tensorflow.org/
TensorFlow Datasets: https://www.tensorflow.org/datasets
