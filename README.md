# Breast Cancer Classification

Machine Learning project for the binary classification of breast cancer tumors as **benign** or **malignant**.

The project uses the **Breast Cancer Wisconsin (Diagnostic)** dataset from the **UCI Machine Learning Repository**. The goal is to compare different supervised learning models and evaluate their ability to classify tumors based on numerical features extracted from digitized images of breast mass cell nuclei.

---

## Project Description

Breast cancer diagnosis is a binary classification problem where each sample must be classified as either:

- **Benign**
- **Malignant**

In this project, several machine learning models are trained and evaluated on the Breast Cancer Wisconsin Diagnostic dataset.

The workflow includes:

- dataset loading from the UCI Machine Learning Repository;
- exploratory data analysis;
- feature visualization using boxplots and histograms;
- feature selection based on distribution similarity and correlation analysis;
- feature scaling;
- training of different classification models;
- model evaluation using classification reports, confusion matrices and ROC curves.

---

## Dataset

The dataset used in this project is:

**Breast Cancer Wisconsin (Diagnostic)**  
Source: UCI Machine Learning Repository  
Dataset ID: `17`

The dataset contains features computed from digitized images of a fine needle aspirate of breast masses. These features describe characteristics of the cell nuclei present in the images.

### Dataset Characteristics

- **Number of instances:** 569
- **Number of features:** 30
- **Feature type:** Real-valued
- **Task:** Binary classification
- **Target variable:** Diagnosis
- **Classes:**
  - `B`: Benign
  - `M`: Malignant

In the code, the target variable is mapped as follows:

```python
B -> 0
M -> 1
