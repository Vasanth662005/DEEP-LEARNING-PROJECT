# DEEP-LEARNING-PROJECT

*COMPANY*: CODTECH IT SOLUTIONS

*NAME*: N VASANTHA KUMAR

*INTERN ID*: CT08DG148

*DOMAIN*: Data Science

*DURATION*: 8 WEEKS

*MENTOR*: NEELA SANTOSH

# Data Science Internship Task 2 — CIFAR-10 Image Classification

**Internship Organization:** CODTECH  
**Intern Name:** N Vasantha Kumar 
**Task Title:** Deep Learning Model for Image Classification  
**Technology Stack:** TensorFlow, Keras, Matplotlib, Python

---

## Objective

The objective of this task was to implement a deep learning model for image classification using either TensorFlow or PyTorch. For this internship task, I chose **image classification using the CIFAR-10 dataset** and implemented the entire pipeline in **TensorFlow with Keras API**.

The CIFAR-10 dataset is a widely-used benchmark dataset consisting of 60,000 32×32 color images in 10 classes, with 6,000 images per class. The goal was to design, train, evaluate, and visualize a Convolutional Neural Network (CNN) that can classify these images accurately.

---

## Problem Statement

Build an image classification model that takes raw image pixel data as input and outputs the class label prediction among the 10 categories in CIFAR-10. This task simulates a common real-world deep learning scenario where visual recognition is performed using CNNs.

---

## Dataset Information

The **CIFAR-10** dataset contains the following 10 categories:

- airplane
- automobile
- bird
- cat
- deer
- dog
- frog
- horse
- ship
- truck

There are 50,000 training images and 10,000 test images.

---

## Tools & Libraries Used

- Python 3.12
- TensorFlow 2.x with Keras API
- Matplotlib for visualization
- NumPy for array handling
- scikit-learn for evaluation (optional)

---

## Project Workflow

### 1. Load and Preprocess Data
- Loaded CIFAR-10 dataset directly from `tensorflow.keras.datasets`
- Normalized image pixel values to the [0, 1] range
- Displayed sample images with their corresponding class labels

### 2. Build CNN Model
- Used 3 convolutional layers with ReLU activations and max-pooling
- Flattened the output and passed it through a dense layer followed by a softmax output layer for multiclass classification

### 3. Compile and Train Model
- Used `Adam` optimizer
- Loss function: `sparse_categorical_crossentropy`
- Metrics: Accuracy
- Trained the model for 10 epochs with validation on the test set

### 4. Evaluate Model
- Plotted training vs. validation accuracy and loss
- Saved the model to `cifar10_model.h5` using Keras

### 5. Prediction Visualization
- Ran predictions on test images
- Visualized 5 sample predictions vs actual labels

---

## Results

The model achieved around **78% training accuracy** and **70% validation accuracy** after 10 epochs. The validation loss and accuracy plots showed good learning behavior, and the model generalized reasonably well on unseen data.

---

## Files Included

| File Name                                | Description                                      |
|------------------------------------------|--------------------------------------------------|
| `Task2_CIFAR10_Image_Classification.ipynb` | Jupyter notebook with full code and explanations |
| `cifar10_model.h5`                       | Saved trained CNN model                          |
| `README.md`                              | Project documentation                            |
| `screenshots/accuracy_loss_plot.png`     | Accuracy & loss curves                           |
| `screenshots/sample_predictions.png`     | Example prediction visualization                 |

---

## How to Run

1. Clone the repository or download the files.
2. Open the notebook `Task2_CIFAR10_Image_Classification.ipynb` in Jupyter or Colab.
3. Run all cells in sequence to train the model and view outputs.
4. Model will be saved as `cifar10_model.h5`.

---

## Learning Outcomes

- Understanding of CNN architecture and hyperparameters
- Experience with TensorFlow and Keras API
- Learned to visualize training metrics using Matplotlib
- Hands-on experience with model evaluation and predictions

---

## Submission

This project is submitted as part of **Task 2 of the CODTECH Data Science Internship**. It demonstrates my ability to build and train a deep learning model using TensorFlow for real-world image classification tasks.

## Output 

![Image](https://github.com/user-attachments/assets/6064830f-2775-4aa6-b436-10ca1c620598)
