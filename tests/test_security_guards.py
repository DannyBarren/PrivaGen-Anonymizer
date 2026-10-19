"""Security helpers that must stay true on a public repo."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from scripts.secrets_manager import is_placeholder
from scripts.security import SecurityViolationError, assert_not_input_raw, redact_secrets_text


def test_input_raw_write_guard(tmp_path: Path) -> None:
    raw = tmp_path / "input_raw"
    raw.mkdir()
    with pytest.raises(SecurityViolationError):
        assert_not_input_raw(raw / "keep.jpg", raw)


def test_redact_openai_style_assignment() -> None:
    redacted = redact_secrets_text("OPENAI_API_KEY=sk-your-openai-key-REPLACE_ME")
    assert "REPLACE_ME" not in redacted
    assert "***REDACTED***" in redacted


def test_placeholder_markers_include_prior_audit_shapes() -> None:
    assert is_placeholder("sk-your-openai-key")
    assert is_placeholder("sk-MOCK-OPENAI-REPLACE_ME")
    assert is_placeholder("your-readonly-application-key")
    assert is_placeholder("")
    assert not is_placeholder("not-a-template-value-at-all")


def test_redact_uses_live_env_value(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "sk-live-looking-but-fake-value")
    os.environ["OPENAI_API_KEY"] = "sk-live-looking-but-fake-value"
    text = redact_secrets_text("header sk-live-looking-but-fake-value trailing")
    assert "sk-live-looking-but-fake-value" not in text
    assert "***REDACTED***" in text
