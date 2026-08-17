# Pneumonia Classification using Deep Learning

This project detects pneumonia from chest X-ray images using deep learning and transfer learning techniques.

The project uses pretrained CNN architectures and evaluates their performance using standard classification metrics.

## Features

* Chest X-ray image preprocessing and validation
* Exploratory data analysis and class distribution analysis
* Image enhancement using CLAHE
* Image augmentation using Albumentations
* Transfer learning with pretrained CNN architectures
* Class-weighted model training
* Early stopping and learning-rate scheduling
* Model evaluation using Accuracy, Precision, Recall, F1-Score and ROC-AUC
* Confusion matrix and ROC curve analysis

## Methodology

1. Loaded and validated chest X-ray images from the dataset.
2. Explored the dataset and analyzed the distribution of Normal and Pneumonia images.
3. Preprocessed and enhanced X-ray images using resizing and CLAHE.
4. Applied image augmentation to improve model generalization.
5. Handled class imbalance using class weights during training.
6. Built transfer-learning models using pretrained VGG16 and ResNet50 architectures.
7. Added custom classification layers with Global Average Pooling and Dropout.
8. Trained the models using Adam optimizer with early stopping and learning-rate reduction.
9. Evaluated model performance using Accuracy, Precision, Recall, F1-Score and ROC-AUC.
10. Compared the CNN models using confusion matrices and ROC curves.

## Models Used

* VGG16
* ResNet50

## Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC

## Technologies Used

* Python
* NumPy
* Pandas
* OpenCV
* Matplotlib
* Seaborn
* Albumentations
* TensorFlow / Keras
* Scikit-learn
