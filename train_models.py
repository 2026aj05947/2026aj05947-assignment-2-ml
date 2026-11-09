"""
ML Assignment 2 - Spam e-mail classification (UCI Spambase)

Trains five classifiers on the same train split, evaluates them on a held-out
test split with six metrics, and saves everything the Streamlit app needs:

    model/<name>.joblib      fitted sklearn pipelines (preprocessing included)
    model/metrics.csv        comparison table on the test split
    model/best_params.json   hyper-parameters chosen by 5-fold CV
    model/meta.json          feature list, target name, library versions
    test_data.csv            held-out test split (features + target)

Run:  python train_models.py
"""
import json
import platform
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

ROOT = Path(__file__).resolve().parent
DATA_FILE = ROOT / "data" / "spambase.csv"
MODEL_DIR = ROOT / "model"
TARGET = "is_spam"
SEED = 2026

# SVC(probability=True) is flagged as deprecated from scikit-learn 1.9; it still
# works and keeps the code compatible with older versions on the lab machine.
warnings.filterwarnings("ignore", category=FutureWarning)
TEST_SIZE = 0.20


def log_scale():
    """log(1+x) to tame the heavy right skew of the frequency / run-length
    columns, followed by standardisation."""
    return [("log1p", FunctionTransformer(np.log1p)),
            ("scale", StandardScaler())]


# name -> (pipeline, small grid searched with 5-fold CV on the training split)
CANDIDATES = {
    "Logistic Regression": (
        Pipeline(log_scale() + [("clf", LogisticRegression(max_iter=5000, random_state=SEED))]),
        {"clf__C": [0.01, 0.1, 1, 10, 100]},
    ),
    "Decision Tree": (
        Pipeline([("clf", DecisionTreeClassifier(random_state=SEED))]),
        {"clf__max_depth": [4, 6, 8, 10, 14, None],
         "clf__min_samples_leaf": [1, 3, 5, 10],
         "clf__criterion": ["gini", "entropy"]},
    ),
    "KNN": (
        Pipeline(log_scale() + [("clf", KNeighborsClassifier())]),
        {"clf__n_neighbors": [3, 5, 7, 9, 11, 15, 21],
         "clf__weights": ["uniform", "distance"]},
    ),
    "Naive Bayes": (
        Pipeline(log_scale() + [("clf", GaussianNB())]),
        {"clf__var_smoothing": [1e-9, 1e-7, 1e-5, 1e-3, 1e-1]},
    ),
    "Support Vector Machine": (
        Pipeline(log_scale() + [("clf", SVC(kernel="rbf", probability=True, random_state=SEED))]),
        {"clf__C": [0.5, 1, 5, 10, 50], "clf__gamma": ["scale", 0.005, 0.01, 0.05]},
    ),
}

FILE_NAMES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}


def evaluate(model, X, y):
    """The six metrics required by the assignment (positive class = spam)."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        "Accuracy": accuracy_score(y, y_pred),
        "AUC": roc_auc_score(y, y_prob),
        "Precision": precision_score(y, y_pred, zero_division=0),
        "Recall": recall_score(y, y_pred, zero_division=0),
        "F1": f1_score(y, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y, y_pred),
    }


def main():
    df = pd.read_csv(DATA_FILE)
    n_raw = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print(f"Loaded {n_raw} rows, {n_raw - len(df)} exact duplicates removed -> {len(df)} rows")

    features = [c for c in df.columns if c != TARGET]
    X, y = df[features], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=SEED)
    print(f"Train: {X_train.shape}  Test: {X_test.shape}  "
          f"spam share train/test: {y_train.mean():.3f}/{y_test.mean():.3f}")

    MODEL_DIR.mkdir(exist_ok=True)
    test_df = X_test.copy()
    test_df[TARGET] = y_test
    test_df.to_csv(ROOT / "test_data.csv", index=False)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    rows, best_params = [], {}
    for name, (pipe, grid) in CANDIDATES.items():
        search = GridSearchCV(pipe, grid, scoring="f1", cv=cv, n_jobs=1)
        search.fit(X_train, y_train)
        model = search.best_estimator_
        joblib.dump(model, MODEL_DIR / FILE_NAMES[name])
        best_params[name] = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        scores = evaluate(model, X_test, y_test)
        rows.append({"ML Model Name": name, **scores})
        print(f"{name:<24} CV-F1={search.best_score_:.4f}  "
              + "  ".join(f"{k}={v:.4f}" for k, v in scores.items())
              + f"   params={best_params[name]}")

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(MODEL_DIR / "metrics.csv", index=False)
    (MODEL_DIR / "best_params.json").write_text(json.dumps(best_params, indent=2, default=str))
    (MODEL_DIR / "meta.json").write_text(json.dumps({
        "dataset": "UCI Spambase (id 94)",
        "target": TARGET,
        "classes": {"0": "Not spam", "1": "Spam"},
        "features": features,
        "models": FILE_NAMES,
        "seed": SEED,
        "test_size": TEST_SIZE,
        "n_rows_after_dedup": len(df),
        "python": platform.python_version(),
        "scikit_learn": sklearn.__version__,
    }, indent=2))

    print("\nComparison table (test split):")
    print(metrics.to_string(index=False))
    print(f"\nSaved models to {MODEL_DIR}/ and test split to test_data.csv")


if __name__ == "__main__":
    main()
