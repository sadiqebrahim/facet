"""Getting images in: browser uploads and image URLs.

A hosted deployment cannot keep the local-path model. Two reasons, and the second is the
serious one:

1. Nobody has a path on the server. "Type the folder your photos are in" only ever made
   sense when the browser and the files were on the same machine.
2. `POST {"paths": ["/etc"]}` was an arbitrary server-side filesystem read for any
   authenticated user, and `/api/index` would happily walk any directory the process could
   see. On one trusted box that is a rough edge. Exposed to the internet it is the whole
   game.

So uploaded bytes land in a directory this module owns, one per account, and every image
row records which account it belongs to. `FACET_ALLOW_SERVER_PATHS=1` restores the old
behaviour for a single-user local install, and it is off by default.
"""
from __future__ import annotations

import hashlib
import ipaddress
import os
import re
import socket
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

#: Per-file ceiling. Big enough for a phone photo, small enough that a hundred of them do
#: not fill a disk while the request is still open.
MAX_FILE_BYTES = 25 * 1024 * 1024
#: Per-account ceiling on stored originals. Enforced before a write, not after.
DEFAULT_QUOTA_BYTES = int(os.environ.get("FACET_USER_QUOTA_MB", "2048")) * 1024 * 1024
MAX_FILES_PER_REQUEST = 200
#: URL fetches are strictly smaller: an unattended fetch of an attacker-chosen URL should
#: not be able to spend 25MB of bandwidth and disk per call.
MAX_URL_BYTES = 12 * 1024 * 1024
URL_TIMEOUT = 12.0
MAX_REDIRECTS = 3

SAFE_NAME = re.compile(r"[^A-Za-z0-9._-]+")

#: Magic numbers, checked before anything decodes the bytes. Extensions and Content-Type
#: are both attacker-supplied and neither says anything about what the file actually is.
MAGIC: tuple[tuple[bytes, str], ...] = (
    (b"\xff\xd8\xff", "jpg"),
    (b"\x89PNG\r\n\x1a\n", "png"),
    (b"BM", "bmp"),
    (b"GIF87a", "gif"),
    (b"GIF89a", "gif"),
)


class UploadError(ValueError):
    """Something about the submitted file is wrong. Always safe to show the user."""


def sniff(data: bytes) -> str | None:
    for magic, ext in MAGIC:
        if data.startswith(magic):
            return ext
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


def safe_name(name: str, fallback: str = "image") -> str:
    stem = SAFE_NAME.sub("_", Path(name or "").name).strip("._") or fallback
    return stem[:80]


@dataclass
class Stored:
    path: Path
    bytes: int
    sha256: str
    width: int
    height: int
    origin_url: str | None = None


class UserStorage:
    """One directory per account, with nothing shared between them.

    Files are named by content hash rather than by the name the browser sent. That makes
    re-uploading the same photo a no-op instead of a duplicate, and it means no filename
    from a request ever reaches the filesystem - path traversal stops being something to
    get right and becomes something that cannot be expressed.
    """

    def __init__(self, root: str | Path, user: str, quota_bytes: int = DEFAULT_QUOTA_BYTES):
        if not user or "/" in user or "\\" in user or user in (".", ".."):
            raise UploadError("invalid account name")
        self.user = user
        self.root = Path(root).resolve() / safe_name(user, "user")
        self.library = self.root / "library"
        self.references = self.root / "references"
        self.quota_bytes = quota_bytes

    def ensure(self) -> None:
        self.library.mkdir(parents=True, exist_ok=True)
        self.references.mkdir(parents=True, exist_ok=True)

    def used_bytes(self) -> int:
        return sum(p.stat().st_size for p in self.root.rglob("*") if p.is_file()) \
            if self.root.exists() else 0

    def save(self, data: bytes, filename: str = "", kind: str = "library",
             origin_url: str | None = None) -> Stored:
        """Validate, decode and store one image. Raises UploadError with a usable message."""
        import cv2
        import numpy as np

        if not data:
            raise UploadError("empty file")
        if len(data) > MAX_FILE_BYTES:
            raise UploadError(
                f"{safe_name(filename)} is {len(data)/1e6:.1f}MB; the limit is "
                f"{MAX_FILE_BYTES/1e6:.0f}MB")
        ext = sniff(data)
        if ext is None:
            raise UploadError(
                f"{safe_name(filename)} is not a JPEG, PNG, WebP, BMP or GIF")

        img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise UploadError(f"{safe_name(filename)} could not be decoded as an image")
        h, w = img.shape[:2]
        if min(h, w) < 32:
            raise UploadError(f"{safe_name(filename)} is too small to hold a face")

        self.ensure()
        used = self.used_bytes()
        if used + len(data) > self.quota_bytes:
            raise UploadError(
                f"that would put you over your {self.quota_bytes/1e6:.0f}MB storage limit "
                f"({used/1e6:.0f}MB used). Remove some images first.")

        digest = hashlib.sha256(data).hexdigest()
        target = (self.library if kind == "library" else self.references) / f"{digest}.{ext}"
        if not target.exists():
            tmp = target.with_suffix(target.suffix + ".part")
            tmp.write_bytes(data)
            os.replace(tmp, target)
        return Stored(target, len(data), digest, w, h, origin_url)

    def remove(self, path: str | Path) -> bool:
        """Delete one stored file, but only if it is genuinely inside this account's tree."""
        p = Path(path)
        try:
            p = p.resolve()
            p.relative_to(self.root)
        except (ValueError, OSError):
            return False
        try:
            p.unlink()
            return True
        except OSError:
            return False

    def purge(self) -> int:
        import shutil
        n = sum(1 for p in self.root.rglob("*") if p.is_file()) if self.root.exists() else 0
        shutil.rmtree(self.root, ignore_errors=True)
        return n


# --------------------------------------------------------------------- URL input

def _is_public_ip(host: str) -> tuple[bool, str]:
    """Resolve a hostname and refuse anything that is not a public unicast address.

    Without this, "fetch an image from a URL" is a request-forgery primitive: the server
    sits inside a network the caller does not, so `http://169.254.169.254/...` or
    `http://10.0.0.5/` would be fetched with the server's own reachability.

    There is a residual TOCTOU window - the name is resolved here and again by the HTTP
    client - which a DNS-rebinding attacker could in principle use. Closing it properly
    means pinning the socket to the validated address; until then this blocks every
    non-adversarial mistake and the ordinary attacker.
    """
    try:
        infos = socket.getaddrinfo(host, None)
    except socket.gaierror:
        return False, f"could not resolve {host}"
    if not infos:
        return False, f"could not resolve {host}"
    for info in infos:
        ip = ipaddress.ip_address(info[4][0])
        if (ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast
                or ip.is_reserved or ip.is_unspecified):
            return False, f"{host} resolves to a non-public address ({ip})"
    return True, ""


def check_url(url: str) -> str:
    u = urlparse((url or "").strip())
    if u.scheme not in ("http", "https"):
        raise UploadError("only http and https URLs are supported")
    if not u.hostname:
        raise UploadError("that URL has no host")
    ok, why = _is_public_ip(u.hostname)
    if not ok:
        raise UploadError(why)
    return u.geturl()


def fetch_url(url: str, max_bytes: int = MAX_URL_BYTES) -> tuple[bytes, str]:
    """Download one image. Returns (bytes, final_url).

    Redirects are followed by hand so that every hop is re-validated: a permitted URL that
    302s to `http://127.0.0.1:8000/api/admin/overview` would otherwise walk straight past
    the check above.
    """
    import requests

    current = check_url(url)
    for _ in range(MAX_REDIRECTS + 1):
        r = requests.get(current, timeout=URL_TIMEOUT, stream=True, allow_redirects=False,
                         headers={"User-Agent": "facet/0.2 (+image import)",
                                  "Accept": "image/*"})
        if r.status_code in (301, 302, 303, 307, 308):
            loc = r.headers.get("location")
            r.close()
            if not loc:
                raise UploadError("that URL redirected to nowhere")
            from urllib.parse import urljoin
            current = check_url(urljoin(current, loc))
            continue
        if r.status_code != 200:
            r.close()
            raise UploadError(f"that URL returned HTTP {r.status_code}")

        declared = r.headers.get("content-length")
        if declared and declared.isdigit() and int(declared) > max_bytes:
            r.close()
            raise UploadError(f"that image is larger than {max_bytes/1e6:.0f}MB")
        buf = bytearray()
        for chunk in r.iter_content(64 * 1024):
            buf += chunk
            if len(buf) > max_bytes:
                r.close()
                raise UploadError(f"that image is larger than {max_bytes/1e6:.0f}MB")
        r.close()
        return bytes(buf), current
    raise UploadError("too many redirects")


def default_upload_root() -> Path:
    return Path(os.environ.get(
        "FACET_UPLOAD_DIR",
        str(Path(__file__).resolve().parents[3] / "data" / "uploads"))).resolve()


def server_paths_allowed() -> bool:
    """Indexing a directory by server-side path. Off unless explicitly turned on."""
    return os.environ.get("FACET_ALLOW_SERVER_PATHS", "").strip().lower() in (
        "1", "true", "yes", "on")


def now() -> float:
    return time.time()
