"""
Shared audio feature extraction, used by both the Colab extraction
notebook and the local Streamlit app so the two never drift apart.
"""

import librosa
import numpy as np


def extract_features(file_path):
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

    return features
