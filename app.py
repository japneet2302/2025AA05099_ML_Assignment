import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    confusion_matrix
)

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("Machine Learning Model Evaluation – Adult Census Dataset")

# --------------------------------------------------
# LOAD TRAINING DATA (adult.csv)
# --------------------------------------------------
@st.cache_data
def load_training_data():
    df = pd.read_csv("adult.csv")
    df["income"] = df["income"].str.strip()
    return df

train_df = load_training_data()

# --------------------------------------------------
# DOWNLOAD SAMPLE TEST CSV
# --------------------------------------------------
st.subheader("Download Sample Test CSV")

with open("test.csv", "rb") as f:
    st.download_button(
        label="Download Test CSV",
        data=f,
        file_name="test.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# UPLOAD TEST DATASET
# --------------------------------------------------
st.subheader("Upload Test Dataset")
uploaded_file = st.file_uploader(
    "Upload CSV file (Adult Census format)",
    type=["csv"]
)

if uploaded_file is None:
    st.warning("Please upload test.csv to evaluate the model.")
    st.stop()

test_df = pd.read_csv(uploaded_file)
st.subheader("Uploaded Dataset Preview")
st.dataframe(test_df.head())

# --------------------------------------------------
# DATA PREPROCESSING FUNCTION
# --------------------------------------------------
def preprocess(df, fit=False, scaler=None):
    df = df.copy()
    df["income"] = df["income"].str.strip()
    y = df["income"].map({"<=50K": 0, ">50K": 1})
    X = df.drop("income", axis=1)

    X = pd.get_dummies(X)

    if fit:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, y, scaler, X.columns
    else:
        X_scaled = scaler.transform(X)
        return X_scaled, y

# --------------------------------------------------
# TRAIN / TEST SPLIT (TEMPORAL SAFE)
# --------------------------------------------------
X_train, y_train, scaler, feature_cols = preprocess(train_df, fit=True)

X_test = test_df.drop("income", axis=1)
X_test = pd.get_dummies(X_test)
X_test = X_test.reindex(columns=feature_cols, fill_value=0)
X_test = scaler.transform(X_test)

y_test = test_df["income"].str.strip().map({"<=50K": 0, ">50K": 1})

# --------------------------------------------------
# MODEL SELECTION
# --------------------------------------------------
st.subheader("Model Selection")

model_name = st.selectbox(
    "Choose a Machine Learning Model",
    [
        "Logistic Regression",
        "KNN",
        "Decision Tree",
        "Random Forest",
        "Naive Bayes",
        "XGBoost"
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB(),
    "XGBoost": XGBClassifier(
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42
    )
}

model = models[model_name]

# --------------------------------------------------
# TRAIN MODEL
# --------------------------------------------------
model.fit(X_train, y_train)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------
y_pred = model.predict(X_test)

if hasattr(model, "predict_proba"):
    y_proba = model.predict_proba(X_test)[:, 1]
else:
    y_proba = y_pred

# --------------------------------------------------
# METRICS
# --------------------------------------------------
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
auc = roc_auc_score(y_test, y_proba)
mcc = matthews_corrcoef(y_test, y_pred)

st.subheader("Model Performance Metrics")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Accuracy", f"{accuracy:.4f}")
col2.metric("Precision", f"{precision:.4f}")
col3.metric("Recall", f"{recall:.4f}")
col4.metric("F1 Score", f"{f1:.4f}")

col5, col6 = st.columns(2)
col5.metric("AUC", f"{auc:.4f}")
col6.metric("MCC", f"{mcc:.4f}")

# --------------------------------------------------
# CONFUSION MATRIX
# --------------------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)

cm_df = pd.DataFrame(
    cm,
    index=["Actual <=50K", "Actual >50K"],
    columns=["Predicted <=50K", "Predicted >50K"]
)

st.dataframe(cm_df)
