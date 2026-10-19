"""
Deterministic typed PII detector for OCR strings, EXIF fields, and text fixtures.

This is the CI-measurable layer of PrivaGen: it classifies *text* that the image
pipeline already extracts (PaddleOCR box text, EXIF/IPTC/XMP payloads). It does
not detect faces or bodies — those stay on InsightFace / DeepPrivacy2 and are
out of scope for the CPU eval.

Design:
- Regex + checksums (Luhn) only. No network, no LLM, no model weights.
- Conservative PERSON / LICENSE_PLATE / GPS rules so precision stays usable.
- Known misses (undashed SSN, bare names, international plates) are documented
  and scored on the labeled corpus rather than papered over.

Entity types
------------
EMAIL, PHONE, SSN, CREDIT_CARD, LICENSE_PLATE, GPS, PERSON, ADDRESS,
IP_ADDRESS, AWS_ACCESS_KEY, API_KEY, DATE_OF_BIRTH
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

ENTITY_TYPES = (
    "EMAIL",
    "PHONE",
    "SSN",
    "CREDIT_CARD",
    "LICENSE_PLATE",
    "GPS",
    "PERSON",
    "ADDRESS",
    "IP_ADDRESS",
    "AWS_ACCESS_KEY",
    "API_KEY",
    "DATE_OF_BIRTH",
)

# OpenAI-style prefixes that are documentation / product names, not keys.
_SK_FALSE_POSITIVES = frozenset(
    {
        "sk-socketio",
        "sk-learn",
        "sk-image",
        "sk-video",
    }
)

_EMAIL = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
)
# US / NANP and +<cc> international with separators. Bare 10-digit runs are a
# known miss (too many order-id collisions).
_PHONE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[-.\s])?(?:\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4})(?!\w)"
)
# Dashed SSN; reject when glued to a larger token (ORD-123-45-6789).
_SSN = re.compile(r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])")
_CARD_RUN = re.compile(r"(?<!\d)(?:\d[ -]?){13,19}(?!\d)")
# US-style plates: ABC-1234, 7XKD221, ABC1234. Misses EU `AB-123-CD`.
_PLATE = re.compile(
    r"(?<![\w-])(?:[A-Z]{3}-?\d{3,4}|\d[A-Z]{3}\d{3}|[A-Z]{2,3}\d{4})(?![\w-])"
)
# Paired decimal degrees with at least 3 fractional digits.
_GPS = re.compile(
    r"(?<![\w.])(-?\d{1,2}\.\d{3,8})\s*,\s*(-?\d{1,3}\.\d{3,8})(?![\w.])"
)
_HONORIFIC_PERSON = re.compile(
    r"\b(?:Mr|Mrs|Ms|Mx|Dr|Prof)\.?\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+)+\b"
)
_LABELED_PERSON = re.compile(
    r"\b(?:Name|Patient|Employee|Driver|Subject)\s*:\s*"
    r"([A-Z][a-z]+(?:\s+[A-Z]\.?)?(?:\s+[A-Z][a-z]+)+)"
)
_ADDRESS = re.compile(
    r"\b\d{1,5}\s+[A-Z][A-Za-z0-9.'-]*(?:\s+[A-Z][A-Za-z0-9.']*)*\s+"
    r"(?:St|Street|Ave|Avenue|Rd|Road|Blvd|Lane|Ln|Dr|Drive|Way|Ct|Court|Apt)"
    r"(?:\.|\b)(?:\s+\d+[A-Za-z]?)?(?:\s*,\s*Apt\s+\d+[A-Za-z]?)?",
    re.IGNORECASE,
)
_IPV4 = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
_AWS = re.compile(r"\bAKIA[0-9A-Z]{16}\b")
# Real-looking sk- keys; exclude Flask-SocketIO / scikit product names.
_API_KEY = re.compile(r"\bsk-(?!SocketIO)[A-Za-z0-9_-]{16,}\b")
_DOB = re.compile(
    r"\b(?:DOB|D\.O\.B\.|date of birth|born)\s*:?\s*"
    r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class Entity:
    entity_type: str
    text: str
    start: int
    end: int
    score: float = 1.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def luhn_ok(digits: str) -> bool:
    """Return True if ``digits`` passes the Luhn checksum."""
    if not digits.isdigit() or not (13 <= len(digits) <= 19):
        return False
    total = 0
    reverse = digits[::-1]
    for i, ch in enumerate(reverse):
        n = ord(ch) - 48
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _add(out: list[Entity], entity_type: str, match: re.Match[str], *, group: int = 0) -> None:
    text = match.group(group)
    start = match.start(group)
    out.append(Entity(entity_type, text, start, start + len(text)))


def _overlaps(a: Entity, b: Entity) -> bool:
    return a.start < b.end and b.start < a.end


def _prefer(existing: Entity, incoming: Entity) -> Entity:
    """When spans overlap, keep the more specific / longer span."""
    rank = {
        "CREDIT_CARD": 6,
        "SSN": 5,
        "AWS_ACCESS_KEY": 5,
        "API_KEY": 5,
        "EMAIL": 4,
        "PHONE": 3,
        "GPS": 3,
        "DATE_OF_BIRTH": 3,
        "LICENSE_PLATE": 2,
        "IP_ADDRESS": 2,
        "ADDRESS": 2,
        "PERSON": 1,
    }
    if rank.get(incoming.entity_type, 0) != rank.get(existing.entity_type, 0):
        return incoming if rank.get(incoming.entity_type, 0) > rank.get(existing.entity_type, 0) else existing
    if (incoming.end - incoming.start) != (existing.end - existing.start):
        return incoming if (incoming.end - incoming.start) > (existing.end - existing.start) else existing
    return existing


def _dedupe(entities: list[Entity]) -> list[Entity]:
    kept: list[Entity] = []
    for ent in sorted(entities, key=lambda e: (e.start, -(e.end - e.start))):
        replaced = False
        next_kept: list[Entity] = []
        for prev in kept:
            if _overlaps(prev, ent):
                next_kept.append(_prefer(prev, ent))
                replaced = True
            else:
                next_kept.append(prev)
        if not replaced:
            next_kept.append(ent)
        # collapse if two previous spans both overlapped incoming
        collapsed: list[Entity] = []
        for item in sorted(next_kept, key=lambda e: (e.start, -e.end)):
            if collapsed and _overlaps(collapsed[-1], item):
                collapsed[-1] = _prefer(collapsed[-1], item)
            else:
                collapsed.append(item)
        kept = collapsed
    return kept


def detect(text: str) -> list[Entity]:
    """Return typed PII spans in ``text`` (may be empty)."""
    if not text:
        return []
    found: list[Entity] = []

    for match in _EMAIL.finditer(text):
        _add(found, "EMAIL", match)

    for match in _PHONE.finditer(text):
        digits = re.sub(r"\D", "", match.group(0))
        if len(digits) < 10:
            continue
        _add(found, "PHONE", match)

    for match in _SSN.finditer(text):
        area = match.group(0).split("-", 1)[0]
        if area in {"000", "666"} or area.startswith("9"):
            continue
        _add(found, "SSN", match)

    for match in _CARD_RUN.finditer(text):
        raw = match.group(0)
        digits = re.sub(r"\D", "", raw)
        if luhn_ok(digits):
            found.append(Entity("CREDIT_CARD", raw.strip(), match.start(), match.start() + len(raw.strip())))

    for match in _PLATE.finditer(text):
        token = match.group(0)
        if token.isdigit():
            continue
        _add(found, "LICENSE_PLATE", match)

    for match in _GPS.finditer(text):
        lat = float(match.group(1))
        lon = float(match.group(2))
        if abs(lat) <= 90 and abs(lon) <= 180:
            _add(found, "GPS", match)

    for match in _HONORIFIC_PERSON.finditer(text):
        _add(found, "PERSON", match)
    for match in _LABELED_PERSON.finditer(text):
        _add(found, "PERSON", match, group=1)

    for match in _ADDRESS.finditer(text):
        _add(found, "ADDRESS", match)

    for match in _IPV4.finditer(text):
        parts = match.group(0).split(".")
        if all(p.isdigit() and 0 <= int(p) <= 255 for p in parts):
            if parts[0] == "0" and parts[1] == "0":
                continue
            _add(found, "IP_ADDRESS", match)

    for match in _AWS.finditer(text):
        _add(found, "AWS_ACCESS_KEY", match)

    for match in _API_KEY.finditer(text):
        token = match.group(0)
        if token.lower() in _SK_FALSE_POSITIVES:
            continue
        _add(found, "API_KEY", match)

    for match in _DOB.finditer(text):
        _add(found, "DATE_OF_BIRTH", match, group=1)

    return _dedupe(found)


def detect_texts(texts: Sequence[str]) -> list[dict[str, Any]]:
    """Detect entities across many strings (OCR boxes, EXIF values)."""
    rows: list[dict[str, Any]] = []
    for idx, raw in enumerate(texts):
        for ent in detect(str(raw or "")):
            row = ent.to_dict()
            row["source_index"] = idx
            rows.append(row)
    return rows


def classify_ocr_boxes(boxes: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Attach typed PII labels to PaddleOCR box dicts (``text`` field)."""
    texts = [str(box.get("text") or "") for box in boxes]
    entities = detect_texts(texts)
    counts: dict[str, int] = {}
    for row in entities:
        counts[row["entity_type"]] = counts.get(row["entity_type"], 0) + 1
    return {"entities": entities, "counts": counts, "source": "ocr"}


def _exif_gps_decimal(gps: Mapping[Any, Any]) -> str | None:
    try:
        import piexif
    except ImportError:
        return None
    try:
        lat = gps.get(piexif.GPSIFD.GPSLatitude)
        lat_ref = gps.get(piexif.GPSIFD.GPSLatitudeRef) or b"N"
        lon = gps.get(piexif.GPSIFD.GPSLongitude)
        lon_ref = gps.get(piexif.GPSIFD.GPSLongitudeRef) or b"E"
        if not lat or not lon:
            return None

        def _ratio(value: Any) -> float:
            if isinstance(value, tuple) and len(value) == 2 and value[1]:
                return float(value[0]) / float(value[1])
            return float(value)

        def _dms(parts: Any) -> float:
            return _ratio(parts[0]) + _ratio(parts[1]) / 60.0 + _ratio(parts[2]) / 3600.0

        lat_v = _dms(lat)
        lon_v = _dms(lon)
        if isinstance(lat_ref, bytes):
            lat_ref = lat_ref.decode("ascii", errors="replace")
        if isinstance(lon_ref, bytes):
            lon_ref = lon_ref.decode("ascii", errors="replace")
        if str(lat_ref).upper().startswith("S"):
            lat_v = -lat_v
        if str(lon_ref).upper().startswith("W"):
            lon_v = -lon_v
        return f"{lat_v:.6f}, {lon_v:.6f}"
    except Exception:  # noqa: BLE001 — EXIF is best-effort
        return None


def iter_exif_strings(path: Path) -> list[str]:
    """Collect human-readable EXIF / GPS strings from an image file."""
    values: list[str] = []
    try:
        from PIL import Image
        from PIL.ExifTags import GPSTAGS, TAGS
    except ImportError:
        return values

    try:
        with Image.open(path) as im:
            exif = im.getexif()
            if not exif:
                gps_ifd: Any = None
            else:
                for tag_id, val in exif.items():
                    name = TAGS.get(tag_id, str(tag_id))
                    if name == "GPSInfo":
                        gps_ifd = val
                        continue
                    if isinstance(val, bytes):
                        try:
                            val = val.decode("utf-8", errors="replace")
                        except Exception:  # noqa: BLE001
                            continue
                    if isinstance(val, str | int | float):
                        values.append(f"{name}: {val}")
                try:
                    gps_ifd = gps_ifd if "gps_ifd" in locals() else exif.get_ifd(0x8825)
                except Exception:  # noqa: BLE001
                    gps_ifd = None
            if gps_ifd:
                pretty = []
                for tag_id, val in dict(gps_ifd).items():
                    pretty.append(f"{GPSTAGS.get(tag_id, tag_id)}={val}")
                values.append("GPS " + " ".join(str(p) for p in pretty))
    except Exception:  # noqa: BLE001
        pass

    try:
        import piexif

        exif_dict = piexif.load(str(path))
        gps = exif_dict.get("GPS") or {}
        decimal = _exif_gps_decimal(gps)
        if decimal:
            values.append(decimal)
        for ifd_name in ("0th", "Exif", "1st"):
            ifd = exif_dict.get(ifd_name) or {}
            for val in ifd.values():
                if isinstance(val, bytes):
                    try:
                        values.append(val.decode("utf-8", errors="replace").strip("\x00"))
                    except Exception:  # noqa: BLE001
                        continue
                elif isinstance(val, str):
                    values.append(val)
    except Exception:  # noqa: BLE001
        pass
    return [v for v in values if v]


def scan_exif_pii(path: Path) -> dict[str, Any]:
    """Detect typed PII inside image metadata (GPS, Artist, UserComment, …)."""
    strings = iter_exif_strings(Path(path))
    entities = detect_texts(strings)
    gps_present = any(row["entity_type"] == "GPS" for row in entities)
    if not gps_present:
        # Raw GPS IFD without decimal serialization still counts as GPS PII.
        try:
            import piexif

            gps = (piexif.load(str(path)).get("GPS") or {})
            if gps.get(piexif.GPSIFD.GPSLatitude) and gps.get(piexif.GPSIFD.GPSLongitude):
                decimal = _exif_gps_decimal(gps) or "0.000, 0.000"
                entities.append(
                    {
                        "entity_type": "GPS",
                        "text": decimal,
                        "start": 0,
                        "end": len(decimal),
                        "score": 1.0,
                        "source_index": len(strings),
                    }
                )
                gps_present = True
        except Exception:  # noqa: BLE001
            pass
    counts: dict[str, int] = {}
    for row in entities:
        counts[row["entity_type"]] = counts.get(row["entity_type"], 0) + 1
    return {"entities": entities, "counts": counts, "gps_present": gps_present, "source": "exif"}


def summarize(entities: Iterable[Mapping[str, Any]]) -> dict[str, int]:
    counts = {name: 0 for name in ENTITY_TYPES}
    for row in entities:
        name = str(row.get("entity_type") or "")
        if name in counts:
            counts[name] += 1
    return counts


def dumps_entities(entities: Sequence[Entity]) -> str:
    return json.dumps([e.to_dict() for e in entities], ensure_ascii=False)
