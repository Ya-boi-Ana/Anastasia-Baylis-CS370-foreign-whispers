import pathlib

from api.src.services import download_engine


def test_download_video_uses_fallback_when_ffmpeg_is_missing(monkeypatch, tmp_path):
    """The downloader should still work if ffmpeg is not installed."""
    monkeypatch.setattr(download_engine, "get_video_info", lambda url: ("VIDEOID12345", "Test Title"))

    created_files = []

    class DummyYDL:
        def __init__(self, opts):
            self.opts = opts

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb):
            return False

        def download(self, urls):
            output = tmp_path / "Test Title.mp4"
            output.write_bytes(b"fake video")
            created_files.append(output)

    monkeypatch.setattr(download_engine, "yt_dlp", download_engine.yt_dlp)
    monkeypatch.setattr(download_engine.yt_dlp, "YoutubeDL", DummyYDL)
    monkeypatch.setattr(download_engine.shutil, "which", lambda name: None)

    output_path = download_engine.download_video(
        "https://www.youtube.com/watch?v=VIDEOID12345",
        str(tmp_path),
        "Test Title",
    )

    assert pathlib.Path(output_path).exists()
    assert pathlib.Path(output_path).name == "Test Title.mp4"
    assert download_engine.yt_dlp.YoutubeDL is DummyYDL
    assert created_files


def test_ytdp_opts_skips_directory_cookiefile(tmp_path, monkeypatch):
    """yt-dlp options should ignore cookies path if it is a directory."""
    cookie_dir = tmp_path / "cookies.txt"
    cookie_dir.mkdir()

    monkeypatch.setenv("YT_COOKIES_FILE", str(cookie_dir))
    opts = download_engine._yt_dlp_opts()

    assert "cookiefile" not in opts
