# CN7050 – Intelligent Systems
# FINAL WORKING VERSION – STABLE PIPELINE

import re
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    confusion_matrix, classification_report, ConfusionMatrixDisplay
)

# =========================================================
# LOAD DATA
# =========================================================
DATA_PATH = Path(__file__).resolve().parent / "coursework_data.csv"
df = pd.read_csv(DATA_PATH)

domain_col = "Domain"
target_col = "category"

df[target_col] = df[target_col].astype(str)

print("Dataset shape:", df.shape)
print(df.head())
print("\nLabel distribution:\n", df[target_col].value_counts())

# =========================================================
# FEATURE EXTRACTION FUNCTION
# =========================================================
def extract_handcrafted(X):
    # X comes as DataFrame with column "Domain"
    X = X.iloc[:, 0].astype(str).str.lower()

    X = X.str.replace(r"^https?://", "", regex=True)
    X = X.str.replace(r"^www\.", "", regex=True)
    X = X.str.split("/").str[0]

    return pd.DataFrame({
        "length": X.str.len(),
        "digits": X.apply(lambda d: sum(c.isdigit() for c in d)),
        "hyphens": X.str.count("-"),
        "dots": X.str.count(r"\."),
        "has_digit": X.apply(lambda d: int(any(c.isdigit() for c in d))),
        "has_hyphen": X.str.contains("-").astype(int),
        "tld": X.apply(lambda d: d.split(".")[-1] if "." in d else "no_tld")
    })

handcrafted_pipeline = Pipeline([
    ("extract", FunctionTransformer(extract_handcrafted, validate=False)),
    ("encode", ColumnTransformer(
        transformers=[
            ("num", "passthrough",
             ["length", "digits", "hyphens", "dots", "has_digit", "has_hyphen"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"), ["tld"])
        ],
        remainder="drop"
    ))
])

# =========================================================
# PREPROCESSOR (ONE ONLY)
# =========================================================
preprocess = ColumnTransformer(
    transformers=[
        ("tfidf", TfidfVectorizer(
            analyzer="char",
            ngram_range=(3, 5),
            min_df=2
        ), domain_col),

        ("handcrafted", handcrafted_pipeline, [domain_col])
    ],
    remainder="drop"
)

# =========================================================
# SPLIT
# =========================================================
X = df[[domain_col]]
y = df[target_col]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Train:", X_train.shape, "Test:", X_test.shape)

# =========================================================
# MODELS
# =========================================================
models = {
    "Logistic Regression": LogisticRegression(max_iter=4000),
    "Linear SVM": LinearSVC(),
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        random_state=42
    )
}

def evaluate(model, name):
    y_pred = model.predict(X_test)

    print(f"\n===== {name} =====")
    print("Accuracy :", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, average="macro", zero_division=0))
    print("Recall   :", recall_score(y_test, y_pred, average="macro", zero_division=0))
    print(classification_report(y_test, y_pred, zero_division=0))

    cm = confusion_matrix(y_test, y_pred, labels=sorted(y_test.unique()))
    disp = ConfusionMatrixDisplay(cm, display_labels=sorted(y_test.unique()))
    disp.plot(values_format="d")
    plt.title(name)
    plt.tight_layout()
    plt.show()

# =========================================================
# BASELINE TRAINING
# =========================================================
for name, clf in models.items():
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("clf", clf)
    ])
    pipe.fit(X_train, y_train)
    evaluate(pipe, f"{name} (Baseline)")

# =========================================================
# TUNING (LOGISTIC REGRESSION)
# =========================================================
print("\n--- TUNING LOGISTIC REGRESSION ---")

logreg_pipe = Pipeline([
    ("preprocess", preprocess),
    ("clf", LogisticRegression(max_iter=4000))
])

logreg_gs = GridSearchCV(
    logreg_pipe,
    {"clf__C": [0.1, 1, 5]},
    cv=5,
    scoring="accuracy",
    n_jobs=-1
)

logreg_gs.fit(X_train, y_train)
evaluate(logreg_gs.best_estimator_, "Logistic Regression (Tuned)")

print("\n✅ FINISHED SUCCESSFULLY")
