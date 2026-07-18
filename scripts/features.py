"""
Shared audio feature extraction, used by both the Colab extraction
notebook and the local Streamlit app so the two never drift apart.
"""

import librosa
import numpy as np


def extract_features(file_path):
    y, sr = librosa.load(file_path, duration=30, sr=22050)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_delta = librosa.feature.delta(mfccs)
    mfcc_delta2 = librosa.feature.delta(mfccs, order=2)

    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    zcr = librosa.feature.zero_crossing_rate(y)
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

    y_harmonic, y_percussive = librosa.effects.hpss(y)
    harmonic_rms = librosa.feature.rms(y=y_harmonic)
    percussive_rms = librosa.feature.rms(y=y_percussive)

    chroma_sync = librosa.util.sync(chroma, beat_frames, aggregate=np.median)

    features = {}
    for i in range(13):
        features[f"mfcc_{i}_mean"] = np.mean(mfccs[i])
        features[f"mfcc_{i}_std"] = np.std(mfccs[i])
        features[f"mfcc_delta_{i}_mean"] = np.mean(mfcc_delta[i])
        features[f"mfcc_delta_{i}_std"] = np.std(mfcc_delta[i])
        features[f"mfcc_delta2_{i}_mean"] = np.mean(mfcc_delta2[i])
        features[f"mfcc_delta2_{i}_std"] = np.std(mfcc_delta2[i])

    features["chroma_mean"] = np.mean(chroma)
    features["chroma_std"] = np.std(chroma)
    features["chroma_sync_mean"] = np.mean(chroma_sync) if chroma_sync.size else np.nan
    features["chroma_sync_std"] = np.std(chroma_sync) if chroma_sync.size else np.nan
    features["spectral_centroid_mean"] = np.mean(spectral_centroid)
    features["spectral_rolloff_mean"] = np.mean(spectral_rolloff)
    features["spectral_bandwidth_mean"] = np.mean(spectral_bandwidth)
    features["zcr_mean"] = np.mean(zcr)
    features["harmonic_rms_mean"] = np.mean(harmonic_rms)
    features["percussive_rms_mean"] = np.mean(percussive_rms)
    features["tempo"] = tempo if np.isscalar(tempo) else tempo[0]

    return {name: float(value) for name, value in features.items()}
