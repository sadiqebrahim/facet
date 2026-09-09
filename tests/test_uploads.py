"""Upload validation, storage isolation and the URL-import guard rails.

The URL importer is the one place in this codebase that makes an outbound request to an
address a user chose, which makes it the one place a request-forgery bug can live. Most of
these tests are about what it must REFUSE.
"""
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from facet.api import uploads  # noqa: E402

JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64


def png_bytes(w=64, h=64):
    cv2 = pytest.importorskip("cv2")
    import numpy as np
    ok, buf = cv2.imencode(".png", np.full((h, w, 3), 128, dtype=np.uint8))
    assert ok
    return buf.tobytes()


# ------------------------------------------------------------------- sniffing

def test_magic_numbers_decide_the_type_not_the_extension():
    assert uploads.sniff(JPEG) == "jpg"
    assert uploads.sniff(PNG) == "png"
    assert uploads.sniff(b"RIFF____WEBPxxxx") == "webp"
    assert uploads.sniff(b"<?php system($_GET[0]); ?>") is None
    assert uploads.sniff(b"") is None


def test_a_script_named_jpg_is_still_refused(tmp_path):
    st = uploads.UserStorage(tmp_path, "u")
    with pytest.raises(uploads.UploadError, match="not a JPEG"):
        st.save(b"#!/bin/sh\nrm -rf /\n", "cat.jpg")


def test_truncated_image_with_valid_magic_is_refused(tmp_path):
    """Magic bytes are necessary, not sufficient - the decoder is the real check."""
    st = uploads.UserStorage(tmp_path, "u")
    with pytest.raises(uploads.UploadError):
        st.save(JPEG, "cat.jpg")


# -------------------------------------------------------------------- storage

def test_files_are_named_by_content_not_by_the_client(tmp_path):
    st = uploads.UserStorage(tmp_path, "u")
    data = png_bytes()
    a = st.save(data, "../../../../etc/passwd", "library")
    assert "passwd" not in str(a.path)
    assert a.path.name == f"{a.sha256}.png"
    assert a.path.parent == st.library
    # the same bytes twice is one file, not two
    b = st.save(data, "other-name.png", "library")
    assert b.path == a.path
    assert len(list(st.library.iterdir())) == 1


def test_accounts_get_separate_directories(tmp_path):
    a, b = uploads.UserStorage(tmp_path, "alice"), uploads.UserStorage(tmp_path, "bob")
    assert a.root != b.root
    a.save(png_bytes(), "x.png", "library")
    assert b.used_bytes() == 0


def test_a_username_cannot_escape_the_upload_root(tmp_path):
    for bad in ("../evil", "a/b", "..", ""):
        with pytest.raises(uploads.UploadError):
            uploads.UserStorage(tmp_path, bad)


def test_quota_is_enforced_before_the_write(tmp_path):
    st = uploads.UserStorage(tmp_path, "u", quota_bytes=1)
    with pytest.raises(uploads.UploadError, match="storage limit"):
        st.save(png_bytes(), "x.png", "library")
    assert not any(st.root.rglob("*.png")) if st.root.exists() else True


def test_remove_refuses_a_path_outside_the_account(tmp_path):
    st = uploads.UserStorage(tmp_path, "u")
    victim = tmp_path / "someone_elses.png"
    victim.write_bytes(png_bytes())
    assert st.remove(victim) is False
    assert victim.exists()


def test_tiny_images_are_refused(tmp_path):
    st = uploads.UserStorage(tmp_path, "u")
    with pytest.raises(uploads.UploadError, match="too small"):
        st.save(png_bytes(8, 8), "x.png", "library")


# ------------------------------------------------------------------ URL guard

@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "ftp://example.com/x.jpg",
    "gopher://example.com/",
    "http://localhost/x.jpg",
    "http://127.0.0.1:8000/api/admin/overview",
    "http://[::1]/x.jpg",
    "http://169.254.169.254/latest/meta-data/",     # cloud instance metadata
    "http://10.0.0.5/x.jpg",
    "http://192.168.1.1/x.jpg",
    "http://172.16.0.1/x.jpg",
    "http://0.0.0.0/x.jpg",
])
def test_private_and_non_http_urls_are_refused(url):
    with pytest.raises(uploads.UploadError):
        uploads.check_url(url)


def test_a_public_url_passes_the_shape_check():
    # No request is made here - check_url only resolves and classifies the address.
    try:
        uploads.check_url("https://example.com/cat.jpg")
    except uploads.UploadError as e:
        # A sandbox with no DNS is allowed to fail, but only for that reason.
        assert "resolve" in str(e)


def test_server_paths_are_off_unless_asked_for(monkeypatch):
    monkeypatch.delenv("FACET_ALLOW_SERVER_PATHS", raising=False)
    assert uploads.server_paths_allowed() is False
    monkeypatch.setenv("FACET_ALLOW_SERVER_PATHS", "1")
    assert uploads.server_paths_allowed() is True
    monkeypatch.setenv("FACET_ALLOW_SERVER_PATHS", "0")
    assert uploads.server_paths_allowed() is False
