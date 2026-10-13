"""音声のキー推定と、2つの音源間のキー差（半音数）の推定。"""

from __future__ import annotations

from dataclasses import dataclass

import librosa
import numpy as np

SAMPLE_RATE = 22050

PITCH_CLASSES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
PITCH_CLASSES_FLAT = ["C", "D♭", "D", "E♭", "E", "F", "G♭", "G", "A♭", "A", "B♭", "B"]

# Krumhansl-Kessler のキープロファイル（C を基準）
MAJOR_PROFILE = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
MINOR_PROFILE = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])


@dataclass
class AudioFeatures:
    chroma: np.ndarray  # 平均クロマ (12,)
    tuning: float  # A440 からのずれ（半音単位, [-0.5, 0.5)）


@dataclass
class KeyEstimate:
    tonic: int  # 0=C ... 11=B
    mode: str  # "major" | "minor"
    confidence: float

    @property
    def name(self) -> str:
        return key_name(self.tonic, self.mode)


@dataclass
class ShiftEstimate:
    semitones: int  # 整数のキー差 [-6, +5]（+6 は -6 と同じ扱い）
    precise: float  # チューニング差を含めた連続値の半音差
    confidence: float  # 1位と2位の相関の差から求めた 0〜1 の信頼度
    similarity: float  # 最良シフト時の相関（同じ曲かどうかの目安）
    scores: list[float]  # シフト -6..+5 それぞれの相関


def key_name(tonic: int, mode: str) -> str:
    # ♭系で書くのが一般的なキーは♭表記にする
    flat_major = {1, 3, 5, 8, 10}
    flat_minor = {0, 2, 5, 7, 10}  # Cm, Dm, Fm, Gm, B♭m
    use_flat = tonic in (flat_major if mode == "major" else flat_minor)
    root = (PITCH_CLASSES_FLAT if use_flat else PITCH_CLASSES)[tonic]
    return root + ("" if mode == "major" else "m")


def parse_key(text: str) -> tuple[int, str]:
    """"C", "F#m", "Bb minor", "E♭" などをパースして (tonic, mode) を返す。"""
    s = text.strip().replace("♯", "#").replace("♭", "b")
    if not s:
        raise ValueError("キーが空です")
    letter = s[0].upper()
    if letter not in "CDEFGAB":
        raise ValueError(f"キーを解釈できません: {text}")
    tonic = PITCH_CLASSES.index(letter)
    rest = s[1:]
    if rest[:1] == "#":
        tonic += 1
        rest = rest[1:]
    elif rest[:1] == "b":
        tonic -= 1
        rest = rest[1:]
    rest = rest.strip().lower()
    if rest in ("", "maj", "major", "メジャー", "長調"):
        mode = "major"
    elif rest in ("m", "min", "minor", "マイナー", "短調"):
        mode = "minor"
    else:
        raise ValueError(f"キーを解釈できません: {text}")
    return tonic % 12, mode


def load_audio(path: str, max_seconds: float | None = None) -> np.ndarray:
    y, _ = librosa.load(path, sr=SAMPLE_RATE, mono=True, duration=max_seconds)
    return y


def extract_features(y: np.ndarray, sr: int = SAMPLE_RATE) -> AudioFeatures:
    if y.size < sr:
        raise ValueError("音声が短すぎます")
    # 打楽器成分を除いて和声成分だけでクロマを取る
    y_h = librosa.effects.harmonic(y)
    tuning = float(librosa.estimate_tuning(y=y_h, sr=sr))
    chroma = librosa.feature.chroma_cqt(y=y_h, sr=sr, tuning=tuning)
    # 無音に近いフレームを除外
    energy = chroma.sum(axis=0)
    mask = energy > np.percentile(energy, 10)
    mean = chroma[:, mask].mean(axis=1) if mask.any() else chroma.mean(axis=1)
    if not np.any(mean > 0):
        raise ValueError("音程成分が検出できませんでした")
    return AudioFeatures(chroma=mean / mean.sum(), tuning=tuning)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def estimate_key(chroma: np.ndarray) -> KeyEstimate:
    scores = []
    for mode, profile in (("major", MAJOR_PROFILE), ("minor", MINOR_PROFILE)):
        for tonic in range(12):
            scores.append((_corr(chroma, np.roll(profile, tonic)), tonic, mode))
    scores.sort(reverse=True)
    best, second = scores[0], scores[1]
    confidence = float(np.clip((best[0] - second[0]) * 5, 0, 1))
    return KeyEstimate(tonic=best[1], mode=best[2], confidence=confidence)


def estimate_shift(original: AudioFeatures, cover: AudioFeatures) -> ShiftEstimate:
    """original を何半音動かすと cover になるかを推定する。"""
    shifts = list(range(-6, 6))
    scores = [_corr(np.roll(original.chroma, k), cover.chroma) for k in shifts]
    order = np.argsort(scores)[::-1]
    best, second = scores[order[0]], scores[order[1]]
    k = shifts[order[0]]
    precise = k + (cover.tuning - original.tuning)
    confidence = float(np.clip((best - second) * 2.5, 0, 1))
    return ShiftEstimate(
        semitones=k,
        precise=float(precise),
        confidence=confidence,
        similarity=float(best),
        scores=[float(s) for s in scores],
    )


def shift_from_keys(original: tuple[int, str], cover: KeyEstimate) -> int:
    """原曲キー（手入力）と推定キーから半音差を求める。

    モードが違う場合は平行調として比較する（例: C と Am は同じ調号）。
    """
    o_tonic, o_mode = original
    c_tonic = cover.tonic
    if o_mode != cover.mode:
        # cover 側を original と同じモードの平行調に寄せる
        c_tonic = (c_tonic + 3) % 12 if cover.mode == "minor" else (c_tonic - 3) % 12
    diff = (c_tonic - o_tonic) % 12
    return diff - 12 if diff >= 6 else diff
