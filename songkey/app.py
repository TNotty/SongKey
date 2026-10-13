"""SongKey: 動画が原曲キーから何半音ずれているかを判定する Web アプリ。"""

from __future__ import annotations

import os
import threading
from collections import OrderedDict
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .analysis import (
    AudioFeatures,
    estimate_key,
    estimate_shift,
    extract_features,
    key_name,
    load_audio,
    parse_key,
    shift_from_keys,
)
from .download import DownloadError, VideoInfo, fetch_audio, new_workdir

STATIC_DIR = Path(__file__).parent / "static"
ANALYZE_SECONDS = float(os.environ.get("SONGKEY_ANALYZE_SECONDS", "240"))
CACHE_SIZE = 32

app = FastAPI(title="SongKey")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

_cache: OrderedDict[str, tuple[VideoInfo, AudioFeatures]] = OrderedDict()
_cache_lock = threading.Lock()


class CompareRequest(BaseModel):
    cover_url: str
    original_url: str | None = None
    original_key: str | None = None


def analyze_url(url: str) -> tuple[VideoInfo, AudioFeatures]:
    key = url.strip()
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    with new_workdir() as workdir:
        info, wav = fetch_audio(url, workdir, ANALYZE_SECONDS)
        features = extract_features(load_audio(wav, ANALYZE_SECONDS))
    with _cache_lock:
        _cache[key] = (info, features)
        while len(_cache) > CACHE_SIZE:
            _cache.popitem(last=False)
    return info, features


def describe_shift(semitones: int) -> str:
    if semitones == 0:
        return "原曲キー"
    if abs(semitones) == 6:
        return "原曲から半音6つずれています（上下どちらとも言えます）"
    direction = "上" if semitones > 0 else "下"
    return f"原曲より半音{abs(semitones)}つ{direction}"


def video_json(info: VideoInfo, features: AudioFeatures) -> dict:
    key = estimate_key(features.chroma)
    return {
        **asdict(info),
        "key": key.name,
        "key_confidence": key.confidence,
        "tuning_cents": round(features.tuning * 100),
    }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/api/compare")
def compare(req: CompareRequest) -> dict:
    original_url = (req.original_url or "").strip()
    original_key = (req.original_key or "").strip()
    if not original_url and not original_key:
        raise HTTPException(400, "原曲のURLか原曲キーのどちらかを指定してください")

    parsed_key = None
    if not original_url:
        try:
            parsed_key = parse_key(original_key)
        except ValueError as e:
            raise HTTPException(400, str(e)) from e

    try:
        cover_info, cover_feat = analyze_url(req.cover_url)
        if original_url:
            orig_info, orig_feat = analyze_url(original_url)
    except DownloadError as e:
        raise HTTPException(422, str(e)) from e
    except ValueError as e:
        raise HTTPException(422, f"音声の解析に失敗しました: {e}") from e

    cover = video_json(cover_info, cover_feat)

    if original_url:
        shift = estimate_shift(orig_feat, cover_feat)
        result = {
            "method": "compare",
            "semitones": shift.semitones,
            "precise": round(shift.precise, 2),
            "confidence": round(shift.confidence, 2),
            "similarity": round(shift.similarity, 3),
            "scores": [{"shift": k, "score": round(s, 4)} for k, s in zip(range(-6, 6), shift.scores)],
            "original": video_json(orig_info, orig_feat),
        }
        if shift.similarity < 0.6:
            result["warning"] = "2つの音源の和声があまり似ていません。同じ曲か確認してください。"
    else:
        cover_key = estimate_key(cover_feat.chroma)
        result = {
            "method": "key",
            "semitones": shift_from_keys(parsed_key, cover_key),
            "precise": None,
            "confidence": round(cover_key.confidence, 2),
            "original": {"key": key_name(*parsed_key)},
        }

    result["cover"] = cover
    result["summary"] = describe_shift(result["semitones"])
    return result
