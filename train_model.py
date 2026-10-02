import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# 1. LOAD DATASET
# ============================================================

DATA_PATH = "dataset/kidney_disease.csv"

print("Loading dataset...")
df = pd.read_csv(DATA_PATH)

print("Dataset loaded successfully.")
print("Dataset shape:", df.shape)


# ============================================================
# 2. CLEAN COLUMN NAMES
# ============================================================

df.columns = df.columns.str.strip()


# ============================================================
# 3. CLEAN TARGET COLUMN
# ============================================================

df["classification"] = (
    df["classification"]
    .astype(str)
    .str.strip()
    .str.lower()
)

# Convert target labels
df["classification"] = df["classification"].replace({
    "ckd": 1,
    "notckd": 0
})

# Explicitly convert target to integer
df["classification"] = pd.to_numeric(
    df["classification"],
    errors="coerce"
).astype("int64")

# ============================================================
# 4. REMOVE ID COLUMN
# ============================================================

if "id" in df.columns:
    df = df.drop(columns=["id"])


# ============================================================
# 5. REMOVE INVALID TARGET ROWS
# ============================================================

df = df[df["classification"].isin([0, 1])]
df["classification"] = df["classification"].astype("int64")

print("Cleaned dataset shape:", df.shape)

print("\nTarget distribution:")
print(df["classification"].value_counts())


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(columns=["classification"])
y = df["classification"]


# ============================================================
# 7. IDENTIFY NUMERICAL AND CATEGORICAL FEATURES
# ============================================================

numeric_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()
categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

print("\nNumerical features:")
print(numeric_features)

print("\nCategorical features:")
print(categorical_features)


# ============================================================
# 8. NUMERICAL PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])


# ============================================================
# 9. CATEGORICAL PREPROCESSING
# ============================================================

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ))
])


# ============================================================
# 10. COMBINE PREPROCESSING
# ============================================================

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])


# ============================================================
# 11. MACHINE LEARNING MODEL
# ============================================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


# ============================================================
# 12. COMPLETE ML PIPELINE
# ============================================================

pipeline = Pipeline([
    ("preprocessing", preprocessor),
    ("model", model)
])


# ============================================================
# 13. TRAIN-TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ============================================================
# 14. TRAIN MODEL
# ============================================================

print("\nTraining Random Forest model...")

pipeline.fit(X_train, y_train)

print("Model training completed.")


# ============================================================
# 15. PREDICTION
# ============================================================

y_pred = pipeline.predict(X_test)


# ============================================================
# 16. EVALUATION
# ============================================================

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    zero_division=0
)

cm = confusion_matrix(y_test, y_pred)


# ============================================================
# 17. DISPLAY RESULTS
# ============================================================

print("\n========================================")
print("MODEL EVALUATION")
print("========================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# 18. SAVE MODEL
# ============================================================

os.makedirs("model", exist_ok=True)

MODEL_PATH = "model/ckd_model.joblib"

joblib.dump(pipeline, MODEL_PATH)

metrics_data = {
    "accuracy": round(accuracy, 4),
    "precision": round(precision, 4),
    "recall": round(recall, 4),
    "f1_score": round(f1, 4)
}

joblib.dump(metrics_data, "model/metrics.joblib")

print("Metrics saved to: model/metrics.joblib")

print("\n========================================")
print("MODEL SAVED SUCCESSFULLY")
print("========================================")
print("Saved to:", MODEL_PATH)