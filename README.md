# Breast Cancer Classification

Machine Learning project for the binary classification of breast cancer tumors as **benign** or **malignant**.

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset from the **UCI Machine Learning Repository**. The goal is to compare different supervised learning models and evaluate their ability to classify tumors based on numerical features extracted from digitized images of breast mass cell nuclei.

---

## Project Description

Breast cancer diagnosis is a binary classification problem where each sample must be classified as either:

- **Benign**
- **Malignant**

In this project, several machine learning models are trained and evaluated on the Breast Cancer Wisconsin Diagnostic dataset (https://archive.ics.uci.edu/dataset/17/breast%2Bcancer%2Bwisconsin%2Bdiagnostic).

The workflow includes:

- dataset loading from the UCI Machine Learning Repository;
- exploratory data analysis;
- feature visualization using boxplots and histograms;
- feature selection based on distribution similarity and correlation analysis;
- feature scaling;
- training of different classification models;
- model evaluation using classification reports, confusion matrices and ROC curves.

---


## Project Structure

The project is currently organized as follows:

```text
Breast-cancer-classification/
│
├── Breast.py      # Main script containing the complete classification pipeline
└── README.md      # Project documentation
