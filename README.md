# Fashion-Trend-Analyser-ml-
# Fashion Trend Analyser - Fashion-MNIST Classification

A deep learning project that trains a Convolutional Neural Network (CNN) to classify fashion items from the Fashion-MNIST dataset. This notebook demonstrates a complete machine learning pipeline including data preparation, model training, evaluation, explainability with Grad-CAM, and model deployment.

## 📋 Project Overview

This project implements a CNN classifier for the Fashion-MNIST dataset, which contains 70,000 grayscale images of 10 different fashion categories. The notebook provides a step-by-step guide from data loading to model inference.

## 🎯 Features

- **Data Preprocessing**: Image normalization, reshaping, and data augmentation
- **CNN Architecture**: Custom CNN with batch normalization and dropout
- **Model Training**: With callbacks for checkpointing, early stopping, and learning rate reduction
- **Model Evaluation**: Comprehensive metrics and visualization
- **Explainable AI**: Grad-CAM implementation for model interpretability
- **Model Deployment**: Save/load functionality for inference

## 📊 Dataset

**Fashion-MNIST** - A modern replacement for the original MNIST dataset, containing:
- 60,000 training images
- 10,000 test images
- 10 fashion categories:
  - T-shirt/top
  - Trouser
  - Pullover
  - Dress
  - Coat
  - Sandal
  - Shirt
  - Sneaker
  - Bag
  - Ankle boot

## 🏗️ Model Architecture

The CNN model includes:
- Multiple Conv2D layers with ReLU activation
- Batch Normalization for stable training
- MaxPooling2D for dimensionality reduction
- Dropout layers for regularization
- Global Average Pooling before final classification
- Dense layers with softmax output

**Total Parameters**: ~75,882

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- TensorFlow 2.19.0
- Required libraries: matplotlib, seaborn, scikit-learn, opencv-python

### Installation
```bash
pip install tensorflow matplotlib seaborn scikit-learn opencv-python
