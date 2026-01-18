import streamlit as st
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="Adult Census Income Prediction", layout="wide")
st.title("Adult Census Income Prediction App")

# --------------------------------------------------
# Download test.csv button
# --------------------------------------------------
with open("test.csv", "rb") as f:
    st.download_button(
        label="Download Sample Test CSV",
        data=f,
        file_name="test.csv",
        mime="text/csv"
    )

# --------------------------------------------------
# Upload CSV
# --------------------------------------------------
st.sidebar.header("Upload Dataset")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file (Adult Census format)",
    type=["csv"]
)

if uploaded_file is None:
    st.info("Please upload adult.csv or test.csv to proceed.")
    st.stop()

# --------------------------------------------------
# Load data
# --------------------------------------------------
data = pd.read_csv(uploaded_file)
st.subheader("Dataset Preview")
st.dataframe(data.head())

# --------------------------------------------------
# Validate target column
# --------------------------------------------------
if "income" not in data.columns:
    st.error(
        "❌ Column 'income' not found.\n\n"
        "For this assignment, the dataset MUST contain the target column "
        "`income` to compute confusion matrix and metrics."
    )
    st.stop()

# --------------------------------------------------
# Preprocessing
# --------------------------------------------------
data.replace("?", np.nan, inplace=True)

for col in data.select_dtypes(include="object").columns:
    data[col].fillna("Unknown", inplace=True)

X = data.drop("income", axis=1)
y = data["income"].map({"<=50K": 0, ">50K": 1})

categorical_cols = X.select_dtypes(include="object").columns
numerical_cols = X.select_dtypes(exclude="object").columns

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ("num", "passthrough", numerical_cols)
    ]
)

# --------------------------------------------------
# Model selection
# --------------------------------------------------
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a Machine Learning Model",
    ["KNN", "Decision Tree", "Random Forest"]
)

if model_choice == "KNN":
    model = KNeighborsClassifier(n_neighbors=5)
elif model_choice == "Decision Tree":
    model = DecisionTreeClassifier(random_state=42)
else:
    model = RandomForestClassifier(n_estimators=100, random_state=42)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("model", model)
])

# --------------------------------------------------
# Train-test split
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Train model
# --------------------------------------------------
pipeline.fit(X_train, y_train)

# --------------------------------------------------
# Predictions
# --------------------------------------------------
y_pred = pipeline.predict(X_test)

# --------------------------------------------------
# Metrics
# --------------------------------------------------
st.subheader("Model Performance Metrics")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy_score(y_test, y_pred):.4f}")
col2.metric("Precision", f"{precision_score(y_test, y_pred):.4f}")
col3.metric("Recall", f"{recall_score(y_test, y_pred):.4f}")
col4.metric("F1 Score", f"{f1_score(y_test, y_pred):.4f}")

# --------------------------------------------------
# Confusion Matrix
# --------------------------------------------------
st.subheader("Confusion Matrix")

cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(
    cm,
    index=["Actual <=50K", "Actual >50K"],
    columns=["Predicted <=50K", "Predicted >50K"]
)

st.dataframe(cm_df)

# --------------------------------------------------
# Prediction Preview
# --------------------------------------------------
st.subheader("Prediction Sample")

sample_preds = pipeline.predict(X.head(10))
sample_probs = pipeline.predict_proba(X.head(10))[:, 1]

preview = X.head(10).copy()
preview["Actual Income"] = y.head(10).map({0: "<=50K", 1: ">50K"})
preview["Predicted Income"] = np.where(sample_preds == 1, ">50K", "<=50K")
preview["Confidence"] = sample_probs.round(3)

st.dataframe(preview)
