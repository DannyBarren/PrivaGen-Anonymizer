"""Unit tests for the typed PII detector (no GPU, no images in git)."""

from __future__ import annotations

from scripts.pii_detect import detect, luhn_ok


def _types(text: str) -> list[str]:
    return [e.entity_type for e in detect(text)]


def test_email_and_plus_tag() -> None:
    hits = detect("write ada+qa@example.org please")
    assert [e.text for e in hits] == ["ada+qa@example.org"]
    assert hits[0].entity_type == "EMAIL"


def test_phone_and_ssn_do_not_collide() -> None:
    text = "SSN 123-45-6789 and phone (415) 555-0134"
    types = _types(text)
    assert types.count("SSN") == 1
    assert types.count("PHONE") == 1


def test_luhn_accepts_published_test_pans() -> None:
    assert luhn_ok("4111111111111111")
    assert luhn_ok("5555555555554444")
    assert not luhn_ok("1234567890123456")


def test_credit_card_spaced() -> None:
    hits = detect("PAN 4111 1111 1111 1111")
    assert any(e.entity_type == "CREDIT_CARD" for e in hits)


def test_api_key_placeholder_not_flask_socketio() -> None:
    text = "Flask-SocketIO is fine; key sk-your-openai-key-REPLACE_ME is not"
    types = _types(text)
    assert "API_KEY" in types
    assert types.count("API_KEY") == 1


def test_order_id_is_not_ssn() -> None:
    assert "SSN" not in _types("Reference ORD-123-45-6789")


def test_undashed_ssn_is_known_miss() -> None:
    assert "SSN" not in _types("raw 123456789 only")


def test_bare_name_is_known_miss() -> None:
    assert "PERSON" not in _types("Ada Lovelace called about the stills")


def test_honorific_person() -> None:
    hits = detect("signed by Dr. Jane Roe today")
    assert any(e.entity_type == "PERSON" and "Jane Roe" in e.text for e in hits)


def test_gps_pair() -> None:
    hits = detect("geotag 37.774900, -122.419400")
    assert any(e.entity_type == "GPS" for e in hits)


def test_aws_example_key() -> None:
    hits = detect("AKIAIOSFODNN7EXAMPLE")
    assert [e.entity_type for e in hits] == ["AWS_ACCESS_KEY"]
