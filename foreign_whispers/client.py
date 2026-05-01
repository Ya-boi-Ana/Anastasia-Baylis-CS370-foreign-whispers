"""SDK client for the Foreign Whispers API."""

from __future__ import annotations

import json as _json
import requests


def _djb2(s: str) -> str:
    h = 5381
    for ch in s:
        h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
    return format(h, "07x")[:7]


def config_id(dubbing: str = "baseline") -> str:
    return "c-" + _djb2(_json.dumps({"d": dubbing}, separators=(",", ":")))


BASELINE = config_id("baseline")
ALIGNED = config_id("aligned")


class FWClient:
    """Synchronous client for the Foreign Whispers API."""

    def __init__(self, base_url: str = "http://localhost:8080") -> None:
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _post(self, path: str, **kwargs) -> dict:
        resp = self._session.post(self._url(path), **kwargs)

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"POST {self._url(path)} failed\n"
                f"Status: {resp.status_code}\n"
                f"Response: {resp.text}"
            ) from exc

        return resp.json()

    def _get_json(self, path: str, **kwargs) -> dict | list:
        resp = self._session.get(self._url(path), **kwargs)

        try:
            resp.raise_for_status()
        except requests.HTTPError as exc:
            raise RuntimeError(
                f"GET {self._url(path)} failed\n"
                f"Status: {resp.status_code}\n"
                f"Response: {resp.text}"
            ) from exc

        return resp.json()

    def healthz(self) -> dict:
        return self._get_json("/healthz")

    def videos(self) -> list[dict]:
        return self._get_json("/api/videos")

    def download(self, url: str) -> dict:
        return self._post("/api/download", json={"url": url})

    def transcribe(self, video_id: str, force: bool = True) -> dict:
        """
        Run Whisper transcription.

        force=True is intentional so the app does not skip transcription
        just because YouTube captions already exist.
        """
        return self._post(
            f"/api/transcribe/{video_id}",
            params={"force": str(force).lower()},
        )

    def translate(
        self,
        video_id: str,
        target_language: str = "es",
    ) -> dict:
        return self._post(
            f"/api/translate/{video_id}",
            params={"target_language": target_language},
        )

    def tts(
        self,
        video_id: str,
        config: str = BASELINE,
        alignment: bool = False,
    ) -> dict:
        return self._post(
            f"/api/tts/{video_id}",
            params={
                "config": config,
                "alignment": str(alignment).lower(),
            },
        )

    def stitch(
        self,
        video_id: str,
        config: str = BASELINE,
    ) -> dict:
        return self._post(
            f"/api/stitch/{video_id}",
            params={"config": config},
        )

    def evaluate(self, video_id: str) -> dict:
        return self._get_json(f"/api/evaluate/{video_id}")

    def eval_align(
        self,
        video_id: str,
        max_stretch: float = 1.4,
    ) -> dict:
        return self._post(
            f"/api/eval/{video_id}",
            json={"max_stretch": max_stretch},
        )

    def run_pipeline(
        self,
        url: str,
        config: str = BASELINE,
        alignment: bool = False,
        target_language: str = "es",
        force_transcribe: bool = True,
    ) -> dict:
        """
        Run full pipeline:

        download → transcribe → translate → tts → stitch
        """

        dl = self.download(url)
        video_id = dl["video_id"]

        tr = self.transcribe(video_id, force=force_transcribe)
        tl = self.translate(video_id, target_language=target_language)
        tt = self.tts(video_id, config=config, alignment=alignment)
        st = self.stitch(video_id, config=config)

        return {
            "video_id": video_id,
            "download": dl,
            "transcribe": tr,
            "translate": tl,
            "tts": tt,
            "stitch": st,
        }

    def rerun_from_video_id(
        self,
        video_id: str,
        config: str = BASELINE,
        alignment: bool = False,
        target_language: str = "es",
        force_transcribe: bool = True,
    ) -> dict:
        """
        Re-run pipeline steps after download already happened.
        Useful when fixing TTS/transcription bugs.
        """

        tr = self.transcribe(video_id, force=force_transcribe)
        tl = self.translate(video_id, target_language=target_language)
        tt = self.tts(video_id, config=config, alignment=alignment)
        st = self.stitch(video_id, config=config)

        return {
            "video_id": video_id,
            "transcribe": tr,
            "translate": tl,
            "tts": tt,
            "stitch": st,
        }

    def __repr__(self) -> str:
        return f"FWClient({self.base_url!r})"