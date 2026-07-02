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
from pathlib import Path

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

DATA_PATH = Path(__file__).parent.parent / "data" / "features_final.csv"
MODEL_DIR = Path(__file__).parent.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    feature_cols = [c for c in df.columns if c not in ("track_id", "genre")]
    X = df[feature_cols].values
    y = df["genre"].values
    return X, y, feature_cols, df


def main():
    print("Loading feature data...")
    X, y, feature_cols, df = load_data()
    print(f"Loaded {len(df)} tracks with {len(feature_cols)} features across {len(set(y))} genres")

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    models = {
        "random_forest": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
        "svm": SVC(kernel="rbf", C=10, probability=True, random_state=42),
    }

    results = {}
    best_model = None
    best_score = 0
    best_name = None

    for name, model in models.items():
        print(f"\nTraining {name}...")
        cv_scores = cross_val_score(model, X_train, y_train, cv=5)
        print(f"  Cross val accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std():.3f})")

        model.fit(X_train, y_train)
        test_acc = model.score(X_test, y_test)
        print(f"  Test accuracy: {test_acc:.3f}")

        results[name] = test_acc
        if test_acc > best_score:
            best_score = test_acc
            best_model = model
            best_name = name

    print(f"\nBest model: {best_name} with test accuracy {best_score:.3f}")
    y_pred = best_model.predict(X_test)
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

    joblib.dump(best_model, MODEL_DIR / "genre_classifier.pkl")
    joblib.dump(scaler, MODEL_DIR / "scaler.pkl")
    joblib.dump(label_encoder, MODEL_DIR / "label_encoder.pkl")
    joblib.dump(feature_cols, MODEL_DIR / "feature_cols.pkl")
    print(f"\nSaved model and preprocessing artifacts to {MODEL_DIR}")


if __name__ == "__main__":
    main()
