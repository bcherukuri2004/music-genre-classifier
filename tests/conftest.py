import numpy as np
import soundfile as sf
import pytest


@pytest.fixture
def synthetic_audio_path(tmp_path):
    """A short synthetic tone + noise clip, standing in for a real song."""
    sr = 22050
    duration_sec = 5
    t = np.arange(sr * duration_sec)

    tone = 0.3 * np.sin(2 * np.pi * 440 * t / sr)
    noise = 0.05 * np.random.randn(len(t))
    y = (tone + noise).astype(np.float32)

    path = tmp_path / "synthetic.wav"
    sf.write(path, y, sr)
    return str(path)
