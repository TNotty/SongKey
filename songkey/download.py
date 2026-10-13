"""yt-dlp で動画の音声を取得する。"""

from __future__ import annotations

import os
import subprocess
import tempfile
from dataclasses import dataclass
from urllib.parse import urlparse

import yt_dlp

from .analysis import SAMPLE_RATE

MAX_VIDEO_SECONDS = 20 * 60


class DownloadError(Exception):
    pass


@dataclass
class VideoInfo:
    url: str
    title: str
    uploader: str | None
    duration: float | None
    thumbnail: str | None


def validate_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise DownloadError("http(s) の動画URLを入力してください")
    return url


def fetch_audio(url: str, workdir: str, max_seconds: float) -> tuple[VideoInfo, str]:
    """動画の音声を取得し、モノラル WAV に変換したパスを返す。"""
    url = validate_url(url)
    opts = {
        "format": "bestaudio/best",
        "outtmpl": os.path.join(workdir, "src.%(ext)s"),
        "noplaylist": True,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if info is None:
                raise DownloadError("動画情報を取得できませんでした")
            if info.get("_type") == "playlist":
                raise DownloadError("プレイリストではなく個別の動画URLを指定してください")
            duration = info.get("duration")
            if duration and duration > MAX_VIDEO_SECONDS:
                raise DownloadError(f"動画が長すぎます（最大 {MAX_VIDEO_SECONDS // 60} 分）")
            info = ydl.process_ie_result(info, download=True)
            src = ydl.prepare_filename(info)
    except yt_dlp.utils.DownloadError as e:
        msg = str(e).removeprefix("ERROR: ").split("; please report")[0]
        raise DownloadError(f"動画を取得できませんでした: {msg}") from e

    if not os.path.exists(src):
        candidates = [f for f in os.listdir(workdir) if f.startswith("src.")]
        if not candidates:
            raise DownloadError("音声ファイルが見つかりませんでした")
        src = os.path.join(workdir, candidates[0])

    wav = os.path.join(workdir, "audio.wav")
    cmd = [
        "ffmpeg", "-nostdin", "-loglevel", "error", "-y", "-i", src,
        "-t", str(max_seconds), "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), wav,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise DownloadError(f"音声の変換に失敗しました: {result.stderr.strip()[:200]}")

    video = VideoInfo(
        url=url,
        title=info.get("title") or url,
        uploader=info.get("uploader") or info.get("channel"),
        duration=duration,
        thumbnail=info.get("thumbnail"),
    )
    return video, wav


def new_workdir() -> tempfile.TemporaryDirectory:
    return tempfile.TemporaryDirectory(prefix="songkey-")
