import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix
)

# --------------------------------------------------
# Page configuration
# --------------------------------------------------
st.set_page_config(
    page_title="ML Assignment 2 - Income Prediction",
    layout="wide"
)

st.title("💼 Income Prediction using Machine Learning")
st.write("Predict whether income exceeds $50K using different ML classification models.")

# --------------------------------------------------
# Download Sample Test CSV
# --------------------------------------------------
st.subheader("⬇️ Download Sample Test CSV")
with open("test.csv", "rb") as f:
    st.download_button(
        label="Download Test CSV",
        data=f,
        file_name="test.csv",
        mime="text/csv"
    )
# --------------------------------------------------
# Sidebar: Upload dataset
# --------------------------------------------------
st.sidebar.header("Upload Test Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file (Adult Census format)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload a CSV file to proceed.")
    st.stop()

# --------------------------------------------------
# Load and preview data
# --------------------------------------------------
data = pd.read_csv(uploaded_file)

st.subheader("📄 Uploaded Dataset Preview")
st.dataframe(data.head())

# --------------------------------------------------
# Data preprocessing
# --------------------------------------------------
data.replace("?", np.nan, inplace=True)

for col in data.select_dtypes(include="object").columns:
    data[col].fillna("Unknown", inplace=True)
    
if "income" in data.columns:
    X = data.drop("income", axis=1)
else:
    X = data.copy()
    
if "income" in data.columns:
    y = data["income"].map({"<=50K": 0, ">50K": 1})
else:
    y = None
num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object"]).columns

# --------------------------------------------------
# Two preprocessors:
# 1. Sparse (default) → for most models
# 2. Dense → ONLY for Naive Bayes
# --------------------------------------------------
sparse_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
    ]
)

dense_preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), num_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols)
    ]
)

# --------------------------------------------------
# Sidebar: Model selection
# --------------------------------------------------
st.sidebar.header("Model Selection")

model_name = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    (
        "Logistic Regression",
        "Decision Tree",
        "KNN",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    )
)

def get_model_and_preprocessor(name):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000), sparse_preprocessor

    elif name == "Decision Tree":
        return DecisionTreeClassifier(random_state=42), sparse_preprocessor

    elif name == "KNN":
        return KNeighborsClassifier(n_neighbors=5), sparse_preprocessor

    elif name == "Naive Bayes":
        # 🔑 IMPORTANT FIX
        return GaussianNB(), dense_preprocessor

    elif name == "Random Forest":
        return RandomForestClassifier(n_estimators=100, random_state=42), sparse_preprocessor

    elif name == "XGBoost":
        return XGBClassifier(
            eval_metric="logloss",
            random_state=42
        ), sparse_preprocessor

# --------------------------------------------------
# Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Model training
# --------------------------------------------------
model, preprocessor = get_model_and_preprocessor(model_name)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)[:, 1]

# --------------------------------------------------
# Evaluation metrics
# --------------------------------------------------
st.subheader("📊 Model Evaluation Metrics")

col1, col2, col3 = st.columns(3)

col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.3f}")
col1.metric("AUC", f"{roc_auc_score(y_test, y_prob):.3f}")

col2.metric("Precision", f"{precision_score(y_test, y_pred):.3f}")
col2.metric("Recall", f"{recall_score(y_test, y_pred):.3f}")

col3.metric("F1 Score", f"{f1_score(y_test, y_pred):.3f}")
col3.metric("MCC", f"{matthews_corrcoef(y_test, y_pred):.3f}")

# --------------------------------------------------
# Confusion matrix
# --------------------------------------------------
st.subheader("🔍 Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual <=50K", "Actual >50K"],
    columns=["Predicted <=50K", "Predicted >50K"]
)

st.dataframe(cm_df)
