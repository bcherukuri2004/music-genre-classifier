"""
Trains and compares genre classifiers on the extracted feature set.

Run this after you have downloaded features_final.csv from Colab/Drive
and placed it in the data folder.

Usage:
    python scripts/train_model.py
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, ConfusionMatrixDisplay
from xgboost import XGBClassifier

DATA_PATH = Path(__file__).parent.parent / "data" / "features_final.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REPORTS_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ("track_id", "genre")]

    nan_mask = df[feature_cols].isna().any(axis=1)
    if nan_mask.any():
        print(f"Dropping {nan_mask.sum()} rows with missing/NaN feature values")
        df = df[~nan_mask].reset_index(drop=True)

    X = df[feature_cols].values
    y = df["genre"].values
    return X, y, feature_cols, df


def main():
    print("Loading feature data...")
    X, y, feature_cols, df = load_data()
    print(f"Loaded {len(df)} tracks with {len(feature_cols)} features across {len(set(y))} genres")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "random_forest": RandomForestClassifier(random_state=42),
        "svm": SVC(kernel="rbf", probability=True, random_state=42),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
        "xgboost": XGBClassifier(random_state=42, eval_metric="mlogloss"),
    }

    param_grids = {
        "random_forest": {
            "n_estimators": [100, 200, 400],
            "max_depth": [10, 20, None],
        },
        "svm": {
            "C": [1, 10, 100],
            "gamma": ["scale", "auto"],
        },
        "gradient_boosting": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        },
        "xgboost": {
            "n_estimators": [100, 200],
            "learning_rate": [0.05, 0.1],
            "max_depth": [3, 5],
        },
    }

    results = {}
    best_model = None
    best_score = 0
    best_name = None

    for name, model in models.items():
        print(f"\nTuning {name}...")
        grid = GridSearchCV(model, param_grids[name], cv=5, n_jobs=-1)
        grid.fit(X_train, y_train)
        print(f"  Best params: {grid.best_params_}")
        print(f"  Best cross val accuracy: {grid.best_score_:.3f}")

        test_acc = grid.score(X_test, y_test)
        print(f"  Test accuracy: {test_acc:.3f}")

        results[name] = test_acc
        if test_acc > best_score:
            best_score = test_acc
            best_model = grid.best_estimator_
            best_name = name

    print(f"\nBest model: {best_name} with test accuracy {best_score:.3f}")
    y_pred = best_model.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    fig, ax = plt.subplots(figsize=(8, 8))
    ConfusionMatrixDisplay.from_predictions(
        y_test, y_pred, display_labels=label_encoder.classes_, xticks_rotation="vertical", ax=ax
    )
    ax.set_title(f"Confusion matrix - {best_name}")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)
    print(f"Saved confusion matrix to {REPORTS_DIR / 'confusion_matrix.png'}")

    joblib.dump(best_model, MODEL_DIR / "genre_classifier.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")
    print(f"\nSaved model and preprocessing artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
