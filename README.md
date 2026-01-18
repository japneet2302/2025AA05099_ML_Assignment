# 📘 Machine Learning Assignment 2  
**M.Tech (AIML ML Assignment 2) – BITS Pilani**

---

## 1. Problem Statement
The objective of this project is to build and evaluate multiple machine learning classification models on a real-world dataset and deploy the models using an interactive Streamlit web application. The goal is to compare the performance of different classifiers using standard evaluation metrics and identify the best-performing model.


---

## 2. Dataset Description
The **Adult Census Income Dataset** from the UCI Machine Learning Repository is used in this assignment. As it is publicly available, the dataset is not included and can be uploaded by the user in the Streamlit app.
The task is to predict whether an individual’s annual income exceeds **$50K** based on demographic and employment-related attributes. Both adult.csv (training reference) and test.csv (evaluation sample) are included in the repository as per examiner instruction.

**Dataset characteristics:**
- Total instances: **32,561**
- Total features: **14**
- Target variable: `income` (Binary: `<=50K`, `>50K`)
- Numerical features: 6  
- Categorical features: 8  

Missing values represented as `"?"` were handled by replacing them with `"Unknown"`.  
Categorical features were encoded using **OneHotEncoding**, and numerical features were scaled using **StandardScaler**.

---

## 3. Models Used and Evaluation Metrics

All models were trained on the same dataset using an identical preprocessing pipeline to ensure fair comparison.

### Evaluation Metrics Used
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

---

### 3.1 Model Performance Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------|----------|-----|-----------|--------|----|-----|
| Logistic Regression | 0.8543 | 0.9043 | 0.7394 | 0.6097 | 0.6683 | 0.5804 |
| Decision Tree | 0.8138 | 0.7493 | 0.6106 | 0.6250 | 0.6177 | 0.4947 |
| KNN | 0.8311 | 0.8525 | 0.6662 | 0.5982 | 0.6304 | 0.5226 |
| Naive Bayes | 0.5375 | 0.7358 | 0.3364 | 0.9644 | 0.4963 | 0.3241 |
| Random Forest (Ensemble) | 0.8501 | 0.8987 | 0.7236 | 0.6110 | 0.6625 | 0.5704 |
| XGBoost (Ensemble) | **0.8695** | **0.9233** | **0.7757** | **0.6441** | **0.7083** | **0.6255** |

---

## 4. Model-wise Observations

| ML Model | Observation about Model Performance |
|--------|-------------------------------------|
| Logistic Regression | Performs as a strong baseline model with high AUC, indicating good class separation. However, recall is moderate due to class imbalance. |
| Decision Tree | Achieves reasonable accuracy but shows lower AUC and MCC, indicating overfitting and reduced generalization. |
| KNN | Sensitive to feature scaling and choice of K; weaker performance on imbalanced data. |
| Naive Bayes | Very high recall but low precision due to independence assumption. |
| Random Forest (Ensemble) | Good bias–variance trade-off with strong overall performance. |
| XGBoost (Ensemble) | Best overall performance across all metrics due to handling complex non-linear relationships effectively. |

---

## 5. Streamlit Web Application
An interactive Streamlit web application was developed and deployed using **Streamlit Community Cloud**.

### Features:
- CSV upload for test data  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix / classification report  

---

## 6. Repository Structure
```
project-folder/
│-- app.py
│-- requirements.txt
│-- README.md
│-- model/
│   ├── model_training.ipynb


## 7. Deployment
The application is deployed on **Streamlit Community Cloud**.  
The live application link is included in the final submission PDF.

---

## 8. Notes
- The assignment was executed on **BITS Virtual Lab**.
- Screenshot proof of execution is included in the submission PDF.
- All work complies with academic integrity guidelines.

---
