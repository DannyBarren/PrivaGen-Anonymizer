"""Synthetic labeled PII documents. No real people, accounts, or client data."""

from __future__ import annotations

from typing import Any

# Each document: id, text, entities[{type, text}]. Spans are resolved by eval.
# Hard negatives have empty entities. Known misses are labeled but the detector
# is expected to fail — they stay in the corpus so recall is honest.

DOCUMENTS: list[dict[str, Any]] = [
    {
        "id": "email_basic",
        "text": "Please send the export to jane.roe@example.net by Friday.",
        "entities": [{"type": "EMAIL", "text": "jane.roe@example.net"}],
    },
    {
        "id": "email_plus_tag",
        "text": "Alias inbox: ada+qa@example.org is for fixtures only.",
        "entities": [{"type": "EMAIL", "text": "ada+qa@example.org"}],
    },
    {
        "id": "email_multi",
        "text": "cc pat.lee@example.com and kim.nguyen@example.net on the ticket.",
        "entities": [
            {"type": "EMAIL", "text": "pat.lee@example.com"},
            {"type": "EMAIL", "text": "kim.nguyen@example.net"},
        ],
    },
    {
        "id": "phone_parens",
        "text": "Desk line is (415) 555-0134 if the VPN drops.",
        "entities": [{"type": "PHONE", "text": "(415) 555-0134"}],
    },
    {
        "id": "phone_e164",
        "text": "Mobile +1 555-010-2288 is a reserved test number.",
        "entities": [{"type": "PHONE", "text": "+1 555-010-2288"}],
    },
    {
        "id": "phone_dashed",
        "text": "Call 212-555-0199 and ask for the night operator.",
        "entities": [{"type": "PHONE", "text": "212-555-0199"}],
    },
    {
        "id": "ssn_labeled",
        "text": "Fixture card prints SSN 123-45-6789 in the corner.",
        "entities": [{"type": "SSN", "text": "123-45-6789"}],
    },
    {
        "id": "ssn_sentence",
        "text": "Do not ship 078-05-1120 on a public sample (SSA published this).",
        "entities": [{"type": "SSN", "text": "078-05-1120"}],
    },
    {
        "id": "card_visa_test",
        "text": "Use test PAN 4111 1111 1111 1111 and any future expiry.",
        "entities": [{"type": "CREDIT_CARD", "text": "4111 1111 1111 1111"}],
    },
    {
        "id": "card_mc_test",
        "text": "Mastercard fixture 5555555555554444 is industry-reserved.",
        "entities": [{"type": "CREDIT_CARD", "text": "5555555555554444"}],
    },
    {
        "id": "plate_standard",
        "text": "Rear bumper shows ABC-1234 in the synthetic render.",
        "entities": [{"type": "LICENSE_PLATE", "text": "ABC-1234"}],
    },
    {
        "id": "plate_compact",
        "text": "OCR fixture PLATE 7XKD221 on a yellow rectangle.",
        "entities": [{"type": "LICENSE_PLATE", "text": "7XKD221"}],
    },
    {
        "id": "gps_decimal",
        "text": "Photo geotag 37.774900, -122.419400 is a well-known public plaza.",
        "entities": [{"type": "GPS", "text": "37.774900, -122.419400"}],
    },
    {
        "id": "address_us",
        "text": "Ship to 123 Elm St, Apt 4B and shred the label after.",
        "entities": [{"type": "ADDRESS", "text": "123 Elm St, Apt 4B"}],
    },
    {
        "id": "person_honorific",
        "text": "The waiver was signed by Dr. Jane Roe on the fixture form.",
        "entities": [{"type": "PERSON", "text": "Dr. Jane Roe"}],
    },
    {
        "id": "person_labeled",
        "text": "Patient: Alan Smithee requested the stills be blurred.",
        "entities": [{"type": "PERSON", "text": "Alan Smithee"}],
    },
    {
        "id": "ip_v4",
        "text": "Camera last registered from 203.0.113.45 (TEST-NET-3).",
        "entities": [{"type": "IP_ADDRESS", "text": "203.0.113.45"}],
    },
    {
        "id": "aws_example",
        "text": "Docs quote AKIAIOSFODNN7EXAMPLE — never a live key.",
        "entities": [{"type": "AWS_ACCESS_KEY", "text": "AKIAIOSFODNN7EXAMPLE"}],
    },
    {
        "id": "api_key_placeholder",
        "text": "Local .env uses OPENAI_API_KEY=sk-your-openai-key-REPLACE_ME",
        "entities": [{"type": "API_KEY", "text": "sk-your-openai-key-REPLACE_ME"}],
    },
    {
        "id": "dob_labeled",
        "text": "Fixture DOB: 01/15/1980 is not a real person.",
        "entities": [{"type": "DATE_OF_BIRTH", "text": "01/15/1980"}],
    },
    {
        "id": "mixed_id_card",
        "text": (
            "Name: Jane Roe  SSN 123-45-6789  phone (555) 010-2288  "
            "email jane.roe@example.net  plate ABC-1234"
        ),
        "entities": [
            {"type": "PERSON", "text": "Jane Roe"},
            {"type": "SSN", "text": "123-45-6789"},
            {"type": "PHONE", "text": "(555) 010-2288"},
            {"type": "EMAIL", "text": "jane.roe@example.net"},
            {"type": "LICENSE_PLATE", "text": "ABC-1234"},
        ],
    },
    {
        "id": "mixed_exif_dump",
        "text": "Artist=Mr. John Public; GPS 40.712800, -74.006000; UserComment=pat.lee@example.com",
        "entities": [
            {"type": "PERSON", "text": "Mr. John Public"},
            {"type": "GPS", "text": "40.712800, -74.006000"},
            {"type": "EMAIL", "text": "pat.lee@example.com"},
        ],
    },
    {
        "id": "neg_no_pii",
        "text": "Batch 00017 processed 32 frames. QA wave 3. Device fallback=cpu.",
        "entities": [],
    },
    {
        "id": "neg_version_numbers",
        "text": "The build number is 2024.10.19 and the wheel is 2.4.1+cu121.",
        "entities": [],
    },
    {
        "id": "neg_flask_socketio",
        "text": "The dashboard uses Flask-SocketIO and scikit-image; those are not API keys.",
        "entities": [],
    },
    {
        "id": "neg_isbn_lookalike",
        "text": "The ISBN-style catalog code 123-45-6789 is not a person identifier here.",
        "entities": [],
        "notes": "Known false-positive risk: dashed triple can look like an SSN.",
    },
    {
        "id": "neg_order_id",
        "text": "Reference ORD-123-45-6789 must not be treated as an SSN.",
        "entities": [],
    },
    {
        "id": "neg_luhn_fail",
        "text": "Random 16-digit run 1234567890123456 fails Luhn and is not a card.",
        "entities": [],
    },
    {
        "id": "miss_ssn_undashed",
        "text": "The raw nine digits 123456789 have no separators.",
        "entities": [{"type": "SSN", "text": "123456789"}],
        "notes": "Known miss: undashed SSN is not detected (too many integer IDs).",
    },
    {
        "id": "miss_bare_name",
        "text": "Ada Lovelace called about the stills — no honorific, no Name: label.",
        "entities": [{"type": "PERSON", "text": "Ada Lovelace"}],
        "notes": "Known miss: bare proper names are not detected.",
    },
    {
        "id": "miss_intl_phone",
        "text": "London test desk 447700900123 has no separators.",
        "entities": [{"type": "PHONE", "text": "447700900123"}],
        "notes": "Known miss: E.164 without + or separators.",
    },
    {
        "id": "miss_eu_plate",
        "text": "French-style plate AB-123-CD is outside the US pattern.",
        "entities": [{"type": "LICENSE_PLATE", "text": "AB-123-CD"}],
        "notes": "Known miss: EU plate grammar.",
    },
    {
        "id": "miss_face_note",
        "text": "A frontal face is present in the photograph (no text encoding of the face).",
        "entities": [],
        "notes": "Faces are not a text entity. Scored only on the GPU residual gate.",
    },
    {
        "id": "phone_dots",
        "text": "Backup number 646.555.0177 is reserved.",
        "entities": [{"type": "PHONE", "text": "646.555.0177"}],
    },
    {
        "id": "address_avenue",
        "text": "Studio at 500 Market Street during the dry run.",
        "entities": [{"type": "ADDRESS", "text": "500 Market Street"}],
    },
    {
        "id": "person_mr",
        "text": "Waiver on file for Mr. John Public (synthetic).",
        "entities": [{"type": "PERSON", "text": "Mr. John Public"}],
    },
    {
        "id": "gps_alt",
        "text": "Second geotag 51.507400, -0.127800 is a public landmark.",
        "entities": [{"type": "GPS", "text": "51.507400, -0.127800"}],
    },
    {
        "id": "api_key_mock",
        "text": "Historical template used OPENAI_API_KEY=sk-MOCK-OPENAI-REPLACE_ME",
        "entities": [{"type": "API_KEY", "text": "sk-MOCK-OPENAI-REPLACE_ME"}],
    },
    {
        "id": "neg_short_sk",
        "text": "Commit message mentions sk-test only; too short to be a key.",
        "entities": [],
    },
    {
        "id": "mixed_ops_note",
        "text": (
            "Operator note: redact 4111-1111-1111-1111 and notify "
            "ops+alerts@example.com if GPS 34.052200, -118.243700 remains."
        ),
        "entities": [
            {"type": "CREDIT_CARD", "text": "4111-1111-1111-1111"},
            {"type": "EMAIL", "text": "ops+alerts@example.com"},
            {"type": "GPS", "text": "34.052200, -118.243700"},
        ],
    },
]


def resolve_document(doc: dict[str, Any]) -> dict[str, Any]:
    """Attach start/end character offsets for each labeled entity."""
    text = str(doc["text"])
    used: list[tuple[int, int]] = []
    resolved: list[dict[str, Any]] = []
    for spec in doc.get("entities") or []:
        needle = str(spec["text"])
        start = 0
        found = None
        while True:
            idx = text.find(needle, start)
            if idx < 0:
                break
            span = (idx, idx + len(needle))
            if span not in used:
                found = span
                break
            start = idx + 1
        if found is None:
            raise ValueError(f"{doc['id']}: labeled text {needle!r} not found")
        used.append(found)
        resolved.append(
            {
                "entity_type": spec["type"],
                "text": needle,
                "start": found[0],
                "end": found[1],
            }
        )
    return {
        "id": doc["id"],
        "text": text,
        "entities": resolved,
        "notes": doc.get("notes") or "",
    }


def iter_resolved() -> list[dict[str, Any]]:
    return [resolve_document(doc) for doc in DOCUMENTS]
