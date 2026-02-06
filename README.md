# 📘 Machine Learning Assignment 2  
**M.Tech (AIML – Machine Learning Assignment 2)**  
**BITS Pilani – Work Integrated Learning Programme**

---

## 1. Problem Statement
The objective of this assignment is to build, evaluate, and compare multiple supervised machine learning classification models on a real-world dataset. The task involves predicting whether an individual earns more than **$50K per year** based on demographic and employment-related attributes.  

The assignment also requires deploying the trained models using an interactive **Streamlit web application**, allowing users to upload datasets, select models, and view evaluation metrics along with confusion matrices.

---

## 2. Dataset Description
The **Adult Census Income Dataset** (UCI Machine Learning Repository) is used for model training and evaluation.

### Files used:
- **adult.csv** → Primary dataset used for training and evaluation  
- **test.csv** → Sample evaluation dataset provided for examiner validation and download via Streamlit  

Both files are included in the GitHub repository as per updated examiner guidelines.

### Dataset characteristics:
- Total records: **32,561**
- Total features: **14**
- Target variable: `income` (`<=50K`, `>50K`)
- Numerical features (6):  
  `age`, `fnlwgt`, `education.num`, `capital.gain`, `capital.loss`, `hours.per.week`
- Categorical features (8):  
  `workclass`, `education`, `marital.status`, `occupation`, `relationship`, `race`, `sex`, `native.country`

### Preprocessing:
- Missing values (`?`) replaced with `"Unknown"`
- Numerical features scaled using **StandardScaler**
- Categorical features encoded using **OneHotEncoder**
- Identical preprocessing pipeline used across all models to ensure fair comparison

---

## 3. Models Used and Evaluation Metrics

### Machine Learning Models Implemented:
- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)

### Evaluation Metrics Used:
- Accuracy  
- AUC (ROC)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

## 3.1 Model Performance Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8543 | 0.9136 | 0.7502 | 0.6218 | 0.6800 | 0.5911 |
| Decision Tree | 0.8152 | 0.7510 | 0.6303 | 0.6232 | 0.6267 | 0.5039 |
| KNN | 0.8341 | 0.8673 | 0.6832 | 0.6218 | 0.6511 | 0.5436 |
| Naive Bayes | 0.6010 | 0.8300 | 0.3795 | 0.9487 | 0.5421 | 0.3876 |
| Random Forest (Ensemble) | 0.8563 | 0.9105 | 0.7490 | 0.6358 | 0.6878 | 0.5986 |
| XGBoost (Ensemble) | **0.8729** | **0.9341** | **0.7896** | **0.6671** | **0.7232** | **0.6453** |

> **Note:**  
> Results are consistent for both `adult.csv` and `test.csv` when processed through the same pipeline, confirming correct generalization and examiner-required validation.

---

## 4. Model-wise Observations

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Strong baseline model with high AUC and stable performance, though recall is moderately affected by class imbalance. |
| Decision Tree | Easy to interpret but shows lower generalization due to overfitting tendencies. |
| KNN | Performance depends heavily on feature scaling and value of K; moderately affected by class imbalance. |
| Naive Bayes | Extremely high recall but poor precision due to independence assumptions between features. |
| Random Forest | Demonstrates good bias–variance trade-off and robust performance across metrics. |
| XGBoost | Best-performing model overall due to effective handling of non-linear relationships and class imbalance. |

---
