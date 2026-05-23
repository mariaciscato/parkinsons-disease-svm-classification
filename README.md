# Parkinson’s Disease Classification using Support Vector Machines (SVM)

Machine learning project using Support Vector Machines (SVM) to classify Parkinson’s disease from biomedical voice measurements. Includes data preprocessing, feature scaling, cross-validation, and model evaluation using Scikit-learn

# Project Overview

Parkinson’s disease (PD) is a progressive neurological disorder that affects movement and speech production. Early detection can support clinical decision-making and patient care.

In this project, machine learning techniques are applied to biomedical voice measurements to classify individuals as healthy or affected by Parkinson’s disease. The dataset contains vocal frequency, amplitude variation, and nonlinear dynamical features extracted from sustained phonations.

The objectives of this project are to:
- Explore and preprocess biomedical voice data
- Train a Support Vector Machine (SVM) classifier
- Compare model performance with and without feature scaling
- Evaluate classification performance using multiple metrics
- Generate predictions on unseen data

This project also demonstrates the importance of preprocessing pipelines, cross-validation, and reproducible machine learning workflows using Scikit-learn.

# Dataset Information

The dataset contains biomedical voice measurements collected from individuals with and without Parkinson’s disease.

- 195 voice recordings
- 31 individuals
- 23 individuals diagnosed with Parkinson’s disease

Target variable:
- `status = 1` → Parkinson’s disease
- `status = 0` → Healthy individual

The dataset includes several categories of acoustic and nonlinear voice measurements such as:
- Fundamental vocal frequencies
- Jitter measurements
- Shimmer measurements
- Harmonics-to-noise ratios
- Nonlinear dynamical complexity measures

These features are relevant because Parkinson’s disease can affect neuromuscular control involved in speech production, leading to detectable alterations in voice signals.

Dataset source:
https://www.kaggle.com/datasets/vikasukani/parkinsons-disease-data-set

# Technologies Used

- Python
- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn

# Methodology

The workflow implemented in this project includes:
- Exploratory Data Analysis (EDA)
- Data preprocessing
- Stratified train-test splitting
- Support Vector Machine (SVM) classification
- Feature scaling using `StandardScaler`
- Cross-validation
- Model evaluation using:
  - Accuracy
  - Confusion matrix
  - Precision
  - Recall
  - F1-score

Two modeling approaches were compared:
1. SVM without feature scaling
2. SVM with feature scaling using a Scikit-learn `Pipeline`

# Results

The project demonstrated strong classification performance for Parkinson’s disease prediction using biomedical voice measurements.

Key observations:
- The SVM classifier achieved high classification accuracy on the test dataset
- Cross-validation provided a more robust estimate of model generalization performance
- Feature scaling influenced model behavior and evaluation metrics
- Confusion matrices and classification reports highlighted the balance between detecting Parkinson’s disease cases and minimizing false classifications

The project also illustrates the importance of empirically evaluating preprocessing strategies rather than assuming systematic improvements.

# Visualizations

The notebook includes several visualizations to support data exploration and model evaluation:
- Class distribution plot
- Confusion matrices
- Cross-validation accuracy comparison
