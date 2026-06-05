# CNN-and-SVM-Image-Classification-comparison
from pathlib import Path

readme = """# SVM vs. YOLO11n Image Classification

This project compares two image classification approaches on a custom wildlife image dataset:

1. **Deep learning approach:** fine-tuned Ultralytics `YOLO11n-cls`
2. **Classical computer vision approach:** HOG feature extraction with a custom one-vs-one SVM classifier

The goal of the project is to evaluate the tradeoffs between modern deep learning and traditional machine learning methods for classifying wildlife and human images.

## Project Overview

The dataset contains four image classes:

- `coyote`
- `dogs`
- `humans`
- `raccoon`

The notebook performs the full machine learning workflow:

- Loads image data from Google Drive
- Balances the dataset across classes
- Splits the data into training, validation, and test sets
- Fine-tunes a YOLO11n classification model
- Tunes model hyperparameters such as epochs and learning rate
- Builds a traditional HOG + SVM classifier from scratch
- Applies data augmentation for the HOG + SVM training set
- Evaluates both approaches using accuracy, precision, recall, F1-score, and confusion matrices
- Compares YOLO11n against HOG + SVM on the final test set
