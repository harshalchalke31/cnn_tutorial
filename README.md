# Classic Convolutional Neural Network (CNN) Implementations

This repository contains several implementations of a classic Convolutional Neural Network (CNN), focusing on foundational concepts and practical applications in Computer Vision (CV). The project explores the basics of CNN architecture and applies it to well-known image datasets to solidify concepts with hands-on experience.

## Files in This Repository

1. **`cnn.py`**: Contains the `ClassicCNN` class, a basic implementation of a CNN, designed to be modular and easy to understand.
2. **`cnn_notes.txt`**: A collection of essential notes and explanations of CNN concepts. This file covers kernel sizes, pooling operations, loss functions, and activation functions, making it a great reference for beginners.
3. **`cnn_tutorial.ipynb`**: A detailed tutorial notebook that applies the CNN model on the MNIST and CIFAR-10 datasets. This notebook includes all preprocessing steps and demonstrates how the architecture can be adapted for different datasets.
4. **`cats_dogs_classifier.ipynb`**: An example notebook applying the `ClassicCNN` model for a binary classification task using the popular Cats vs Dogs dataset. The dataset is sourced from Kaggle, and a link placeholder is provided for downloading the data.

## Project Overview

This project serves as a beginner-friendly introduction to the fundamentals of Convolutional Neural Networks (CNNs). We implemented and tested the CNN on various image classification tasks to gain practical experience with:

- **MNIST**: Handwritten digit recognition (10 classes).
- **CIFAR-10**: Object recognition (10 classes, including cars, animals, etc.).
- **Cats vs Dogs**: Binary classification problem (cats vs. dogs).

### Architecture Overview

The architecture follows a classic CNN design:
1. **Input Layer**: Image data (2D arrays for black-and-white images or 3D arrays for color images).
2. **Convolutional Layers**: 
   - Multiple kernels (typically 3x3) perform the convolution operation to extract features such as edges, textures, and shapes.
   - Kernels are designed to detect different patterns using various feature detectors (e.g., edge detection, sharpening, etc.).
3. **Activation Function**: 
   - Non-linear activation functions (like ReLU) are applied to introduce non-linearity, essential for capturing complex patterns.
4. **Pooling Layers**:
   - Max pooling and average pooling are used to reduce the spatial dimensions and retain important features, helping the network become spatially invariant.
5. **Fully Connected Layer**:
   - The final feature map is flattened and fed into fully connected layers for classification.
6. **Output Layer**: 
   - Softmax activation for multi-class problems or sigmoid for binary classification.

### Preprocessing Techniques

Each dataset requires specific preprocessing steps:
- **MNIST**: Images are normalized and reshaped to fit the CNN input.
- **CIFAR-10**: The dataset undergoes normalization and one-hot encoding of labels.
- **Cats vs Dogs**: Images are resized to a uniform shape, normalized, and split into training and validation sets.

## How to Run the Project

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/cnn-classic.git
   cd cnn-classic

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

3. Download necessary datasets from Kaggle

4. Run the tutorial notebook for MNIST and CIFAR-10:
   ```bash
   jupyter notebook cnn_tutorial.ipynb

5. Run the Cats vs Dogs classifier notebook:
   ```bash
   jupyter notebook cats_dogs_classifier.ipynb

## Concepts Covered

- **Convolutional Operations**: Using kernels to extract image features such as edges, textures, and shapes.
- **Activation Functions**: Importance of ReLU and its variants in CNNs.
- **Pooling**: The significance of max and average pooling in down-sampling feature maps while preserving important information.
- **Flattening**: The process of converting pooled feature maps into vectors for classification.
- **Loss Functions**: Cross-entropy loss for multi-class and binary classification tasks.
- **Model Evaluation**: Understanding accuracy, loss, and confusion matrices for performance assessment.

## Intuition Behind the Project

The motivation for this project is to solidify foundational knowledge of CNNs by applying them to real-world datasets. By training on MNIST, CIFAR-10, and Cats vs Dogs datasets, we aim to:
1. Gain hands-on experience in training CNNs.
2. Explore the impact of different architectural choices (e.g., kernel sizes, pooling strategies).
3. Understand how CNNs can be adapted for various image classification tasks.

## Conclusion

This project serves as an entry point for beginners who want to get started with CNNs and computer vision. By working through these notebooks and understanding the code, you'll gain practical knowledge that will help in more complex CV tasks in the future.

---

### Future Work

We plan to extend this implementation by introducing:
- **Data Augmentation**: Adding rotation, flipping, and zooming to enhance model generalization.
- **Transfer Learning**: Applying pre-trained models for more complex datasets.
  
Feel free to contribute to this repository or raise issues if you encounter any problems!

---

