"""HTTP-agnostic service wrapping download engine functions."""

import json
import pathlib
from pathlib import Path

from api.src.services import download_engine as _dl


def get_video_info(url: str):
    return _dl.get_video_info(url)


def dv_download_video(
    url: str,
    destination: str,
    filename: str | None = None,
    video_info: tuple[str, str] | None = None,
):
    return _dl.download_video(url, destination, filename, video_info)


def dv_download_caption(
    url: str,
    destination: str,
    filename: str | None = None,
    video_info: tuple[str, str] | None = None,
):
    return _dl.download_caption(url, destination, filename, video_info)


class DownloadService:
    """Thin wrapper around root-level download_video helpers.

    Takes *ui_dir* via constructor so the caller controls where files land.
    """

    def __init__(self, ui_dir: Path) -> None:
        self.ui_dir = ui_dir

    # ------------------------------------------------------------------
    # Delegates
    # ------------------------------------------------------------------

    def get_video_info(self, url: str) -> tuple[str, str]:
        """Return (video_id, title) for a YouTube URL."""
        return get_video_info(url)

    def download_video(
        self,
        url: str,
        destination: str,
        filename: str | None = None,
        video_info: tuple[str, str] | None = None,
    ) -> str:
        """Download an MP4 and return the saved path."""
        if filename is None:
            if video_info is None:
                return dv_download_video(url, destination)
            return dv_download_video(url, destination, video_info=video_info)
        if video_info is None:
            return dv_download_video(url, destination, filename)
        return dv_download_video(url, destination, filename, video_info=video_info)

    def download_caption(
        self,
        url: str,
        destination: str,
        filename: str | None = None,
        video_info: tuple[str, str] | None = None,
    ) -> str:
        """Download captions and return the saved path."""
        if filename is None:
            if video_info is None:
                return dv_download_caption(url, destination)
            return dv_download_caption(url, destination, video_info=video_info)
        if video_info is None:
            return dv_download_caption(url, destination, filename)
        return dv_download_caption(url, destination, filename, video_info=video_info)

    # ------------------------------------------------------------------
    # Helpers (moved from router)
    # ------------------------------------------------------------------

    @staticmethod
    def read_caption_segments(caption_path: pathlib.Path) -> list[dict]:
        """Read line-delimited JSON caption file into a list of segment dicts."""
        segments: list[dict] = []
        if caption_path.exists():
            for line in caption_path.read_text().splitlines():
                line = line.strip()
                if line:
                    segments.append(json.loads(line))
        return segments
