"""
Streamlit app for real time genre prediction and similar track discovery.

This is the only part of the project that runs on your local machine, and
it is lightweight: it loads a small precomputed feature CSV and a trained
model, then extracts features from just the one file you upload. No bulk
audio processing happens here.

Usage:
    streamlit run app/app.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import librosa

from scripts.recommender import TrackRecommender

MODEL_DIR = Path(__file__).parent.parent / "models"
DATA_PATH = Path(__file__).parent.parent / "data" / "features_final.csv"


@st.cache_resource
def load_model():
    model = joblib.load(MODEL_DIR / "genre_classifier.pkl")
    scaler = joblib.load(MODEL_DIR / "scaler.pkl")
    label_encoder = joblib.load(MODEL_DIR / "label_encoder.pkl")
    feature_cols = joblib.load(MODEL_DIR / "feature_cols.pkl")
    return model, scaler, label_encoder, feature_cols


@st.cache_resource
def load_recommender():
    return TrackRecommender(DATA_PATH)


def extract_features_single(file_path, feature_cols):
    y, sr = librosa.load(file_path, duration=30, sr=22050)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

    features = {}
    for i in range(13):
        features[f"mfcc_{i}_mean"] = np.mean(mfccs[i])
        features[f"mfcc_{i}_std"] = np.std(mfccs[i])

    features["chroma_mean"] = np.mean(chroma)
    features["chroma_std"] = np.std(chroma)
    features["spectral_centroid_mean"] = np.mean(spectral_centroid)
    features["spectral_rolloff_mean"] = np.mean(spectral_rolloff)
    features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)
    features["zcr_mean"] = np.mean(zcr)
    features["tempo"] = float(tempo) if np.isscalar(tempo) else float(tempo[0])

    return {c: features[c] for c in feature_cols}


st.set_page_config(page_title="Music Genre Classifier", page_icon="🎵", layout="centered")

st.title("Music Genre Classifier and Recommender")
st.write("Upload a track to predict its genre and find similar tracks in the dataset.")

if not (MODEL_DIR / "genre_classifier.pkl").exists():
    st.error("No trained model found. Run scripts/train_model.py first.")
    st.stop()

model, scaler, label_encoder, feature_cols = load_model()
recommender = load_recommender()

uploaded_file = st.file_uploader("Upload an audio file (mp3 or wav)", type=["mp3", "wav"])

if uploaded_file is not None:
    with st.spinner("Analyzing audio..."):
        suffix = Path(uploaded_file.name).suffix
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(uploaded_file.getbuffer())
            temp_path = Path(tmp.name)

        try:
            feats = extract_features_single(str(temp_path), feature_cols)
        finally:
            temp_path.unlink()

    st.audio(uploaded_file)

    feature_vector = np.array([[feats[c] for c in feature_cols]])
    feature_vector_scaled = scaler.transform(feature_vector)

    prediction = model.predict(feature_vector_scaled)[0]
    probabilities = model.predict_proba(feature_vector_scaled)[0]
    predicted_genre = label_encoder.inverse_transform([prediction])[0]

    st.subheader(f"Predicted genre: {predicted_genre}")

    prob_df = pd.DataFrame({
        "genre": label_encoder.classes_,
        "confidence": probabilities,
    }).sort_values("confidence", ascending=False)
    st.bar_chart(prob_df.set_index("genre"))

    st.subheader("Similar tracks")
    method = st.radio("Similarity method", ["knn", "cosine"], horizontal=True)
    recommendations = recommender.recommend_by_features(feats, n=5, method=method)
    rec_df = pd.DataFrame(recommendations)
    st.dataframe(rec_df, use_container_width=True)

else:
    st.info("Upload a track above to get started.")
