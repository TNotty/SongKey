import os

import soundfile as sf
from fastapi.testclient import TestClient

from songkey import app as app_module
from songkey.analysis import SAMPLE_RATE
from songkey.download import DownloadError, VideoInfo, validate_url

from .test_analysis import song

SHIFTS = {"https://example.com/orig": 0, "https://example.com/cover": 3}


def fake_fetch(url, workdir, max_seconds):
    validate_url(url)
    if url not in SHIFTS:
        raise DownloadError("動画を取得できませんでした")
    path = os.path.join(workdir, "audio.wav")
    sf.write(path, song(SHIFTS[url]), SAMPLE_RATE)
    return VideoInfo(url=url, title=url, uploader="tester", duration=21, thumbnail=None), path


def client(monkeypatch):
    monkeypatch.setattr(app_module, "fetch_audio", fake_fetch)
    app_module._cache.clear()
    return TestClient(app_module.app)


def test_compare_with_original_url(monkeypatch):
    res = client(monkeypatch).post(
        "/api/compare",
        json={"cover_url": "https://example.com/cover", "original_url": "https://example.com/orig"},
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["semitones"] == 3
    assert data["summary"] == "原曲より半音3つ上"
    assert data["original"]["key"] == "C"
    assert data["cover"]["key"] == "E♭"
    assert len(data["scores"]) == 12


def test_compare_with_original_key(monkeypatch):
    res = client(monkeypatch).post(
        "/api/compare", json={"cover_url": "https://example.com/cover", "original_key": "F"}
    )
    assert res.status_code == 200, res.text
    data = res.json()
    assert data["semitones"] == -2
    assert data["summary"] == "原曲より半音2つ下"


def test_requires_original(monkeypatch):
    res = client(monkeypatch).post("/api/compare", json={"cover_url": "https://example.com/cover"})
    assert res.status_code == 400


def test_bad_key(monkeypatch):
    res = client(monkeypatch).post(
        "/api/compare", json={"cover_url": "https://example.com/cover", "original_key": "Z"}
    )
    assert res.status_code == 400


def test_bad_url(monkeypatch):
    c = client(monkeypatch)
    res = c.post("/api/compare", json={"cover_url": "file:///etc/passwd", "original_key": "C"})
    assert res.status_code == 422
    res = c.post("/api/compare", json={"cover_url": "https://example.com/missing", "original_key": "C"})
    assert res.status_code == 422


def test_index(monkeypatch):
    res = client(monkeypatch).get("/")
    assert res.status_code == 200
    assert "SongKey" in res.text
