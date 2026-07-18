import math

from scripts.features import extract_features

EXPECTED_FEATURE_COUNT = 89


def test_extract_features_returns_expected_keys(synthetic_audio_path):
    features = extract_features(synthetic_audio_path)

    assert len(features) == EXPECTED_FEATURE_COUNT
    for i in range(13):
        assert f"mfcc_{i}_mean" in features
        assert f"mfcc_{i}_std" in features
        assert f"mfcc_delta_{i}_mean" in features
        assert f"mfcc_delta_{i}_std" in features
        assert f"mfcc_delta2_{i}_mean" in features
        assert f"mfcc_delta2_{i}_std" in features

    for key in [
        "chroma_mean", "chroma_std", "chroma_sync_mean", "chroma_sync_std",
        "spectral_centroid_mean", "spectral_rolloff_mean", "spectral_bandwidth_mean",
        "zcr_mean", "harmonic_rms_mean", "percussive_rms_mean", "tempo",
    ]:
        assert key in features


def test_extract_features_values_are_finite_numbers(synthetic_audio_path):
    features = extract_features(synthetic_audio_path)

    for name, value in features.items():
        assert isinstance(value, float), f"{name} is not a float: {type(value)}"
        assert math.isfinite(value), f"{name} is not finite: {value}"


def test_tempo_is_positive(synthetic_audio_path):
    features = extract_features(synthetic_audio_path)
    assert features["tempo"] > 0


def test_extract_features_is_deterministic(synthetic_audio_path):
    first = extract_features(synthetic_audio_path)
    second = extract_features(synthetic_audio_path)
    assert first == second
