import numpy as np
import pytest

from songkey.analysis import (
    SAMPLE_RATE,
    KeyEstimate,
    estimate_key,
    estimate_shift,
    extract_features,
    key_name,
    parse_key,
    shift_from_keys,
)

# I-V-vi-IV と ii-V-I を混ぜた、キーがはっきりした進行（C メジャー基準, MIDI ノート）
PROGRESSION = [
    [48, 60, 64, 67],  # C
    [43, 59, 62, 67],  # G
    [45, 60, 64, 69],  # Am
    [41, 60, 65, 69],  # F
    [50, 62, 65, 69],  # Dm
    [43, 59, 62, 65],  # G7
    [48, 60, 64, 67],  # C
]
MELODY = [72, 74, 76, 77, 79, 77, 76, 74, 72, 71, 72, 76, 79, 72]


def tone(midi: float, seconds: float, sr: int = SAMPLE_RATE) -> np.ndarray:
    t = np.arange(int(seconds * sr)) / sr
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    y = sum(np.sin(2 * np.pi * f0 * h * t) / h for h in range(1, 5))
    env = np.minimum(1, np.minimum(t / 0.02, (seconds - t) / 0.05))
    return y * env


def song(transpose: float = 0.0, chord_sec: float = 1.0, repeats: int = 3) -> np.ndarray:
    parts = []
    for _ in range(repeats):
        for i, chord in enumerate(PROGRESSION):
            y = sum(tone(n + transpose, chord_sec) for n in chord) * 0.3
            half = len(y) // 2
            y[:half] += tone(MELODY[(2 * i) % len(MELODY)] + transpose, chord_sec / 2)[:half]
            y[half:] += tone(MELODY[(2 * i + 1) % len(MELODY)] + transpose, chord_sec / 2)[: len(y) - half]
            parts.append(y)
    y = np.concatenate(parts)
    return (y / np.abs(y).max() * 0.8).astype(np.float32)


@pytest.fixture(scope="module")
def original():
    return extract_features(song(0))


@pytest.mark.parametrize("shift", [-5, -3, -1, 0, 2, 4, 5])
def test_estimate_shift_integer(original, shift):
    cover = extract_features(song(shift))
    est = estimate_shift(original, cover)
    assert est.semitones == shift
    assert abs(est.precise - shift) < 0.2
    assert est.similarity > 0.9
    assert est.confidence > 0.2


def test_estimate_shift_fractional(original):
    # 再生速度を上げたような半端なずれ（+1.4 半音）
    cover = extract_features(song(1.4))
    est = estimate_shift(original, cover)
    assert est.semitones == 1
    assert abs(est.precise - 1.4) < 0.2


def test_estimate_key(original):
    key = estimate_key(original.chroma)
    assert (key.tonic, key.mode) == (0, "major")
    key = estimate_key(extract_features(song(3)).chroma)
    assert key.name == "E♭"


@pytest.mark.parametrize(
    "text,expected",
    [
        ("C", (0, "major")),
        ("F#m", (6, "minor")),
        ("B♭", (10, "major")),
        ("Eb minor", (3, "minor")),
        ("Cb", (11, "major")),
        ("a", (9, "major")),
        ("Aマイナー", (9, "minor")),
    ],
)
def test_parse_key(text, expected):
    assert parse_key(text) == expected


@pytest.mark.parametrize("text", ["", "H", "C dorian", "X#"])
def test_parse_key_invalid(text):
    with pytest.raises(ValueError):
        parse_key(text)


def test_key_name():
    assert key_name(1, "major") == "D♭"
    assert key_name(6, "major") == "F#"
    assert key_name(10, "minor") == "B♭m"
    assert key_name(1, "minor") == "C#m"


def test_shift_from_keys():
    assert shift_from_keys((0, "major"), KeyEstimate(2, "major", 1)) == 2
    assert shift_from_keys((0, "major"), KeyEstimate(10, "major", 1)) == -2
    assert shift_from_keys((0, "major"), KeyEstimate(6, "major", 1)) == -6
    # 平行調（Am は C と同じ調号）
    assert shift_from_keys((0, "major"), KeyEstimate(9, "minor", 1)) == 0
    assert shift_from_keys((9, "minor"), KeyEstimate(2, "major", 1)) == 2
