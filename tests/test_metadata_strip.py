"""GPS / EXIF PII is detected before strip and gone after (no committed images)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import piexif
from PIL import Image
from scripts.pii_detect import scan_exif_pii
from scripts.utils import strip_all_metadata


def _gps_rational(deg: float) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    d = int(deg)
    minutes_full = (deg - d) * 60
    m = int(minutes_full)
    s = (minutes_full - m) * 60
    return (d, 1), (m, 1), (int(s * 1000), 1000)


def _write_tagged_jpeg(path: Path) -> None:
    image = Image.new("RGB", (64, 64), color=(12, 24, 36))
    gps = {
        piexif.GPSIFD.GPSLatitudeRef: b"N",
        piexif.GPSIFD.GPSLatitude: _gps_rational(37.7749),
        piexif.GPSIFD.GPSLongitudeRef: b"W",
        piexif.GPSIFD.GPSLongitude: _gps_rational(122.4194),
    }
    zeroth = {
        piexif.ImageIFD.Artist: b"Ada Lovelace",
        piexif.ImageIFD.Make: b"SyntheticCam",
    }
    exif = {
        piexif.ExifIFD.UserComment: b"contact jane.roe@example.net",
    }
    exif_bytes = piexif.dump({"0th": zeroth, "Exif": exif, "GPS": gps})
    image.save(path, format="JPEG", exif=exif_bytes)


def test_exif_scan_finds_gps_and_email() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        src = Path(tmp) / "tagged.jpg"
        _write_tagged_jpeg(src)
        scan = scan_exif_pii(src)
        assert scan["gps_present"] is True
        types = {row["entity_type"] for row in scan["entities"]}
        assert "GPS" in types
        assert "EMAIL" in types


def test_strip_all_metadata_removes_gps(tmp_path: Path) -> None:
    src = tmp_path / "tagged.jpg"
    out = tmp_path / "clean.jpg"
    _write_tagged_jpeg(src)
    result = strip_all_metadata(src, out, prefer_exiftool=False)
    assert result["method"] == "pillow_reencode"
    after = scan_exif_pii(out)
    assert after["gps_present"] is False
    assert after["counts"].get("GPS", 0) == 0
    # Original bytes stay tagged — strip writes a new file.
    assert scan_exif_pii(src)["gps_present"] is True
