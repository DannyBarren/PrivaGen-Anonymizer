"""Eval floors — measured quality, not a perfect-score assertion."""

from __future__ import annotations

from scripts.eval_pii import evaluate


def test_eval_reports_every_labeled_type() -> None:
    report = evaluate()
    assert report["documents"] >= 30
    assert report["gold_entities"] >= 25
    labeled = {
        name
        for name, row in report["per_entity_type"].items()
        if row["support"] > 0
    }
    assert {"EMAIL", "PHONE", "SSN", "CREDIT_CARD", "GPS", "PERSON"}.issubset(labeled)


def test_high_precision_types_meet_floors() -> None:
    """Well-structured identifiers should be reliable enough to ship."""
    by_type = evaluate()["per_entity_type"]
    for name in ("EMAIL", "CREDIT_CARD", "AWS_ACCESS_KEY", "API_KEY", "GPS"):
        row = by_type[name]
        assert row["support"] >= 1, name
        assert row["precision"] >= 0.9, (name, row)
        assert row["recall"] >= 0.9, (name, row)


def test_micro_quality_is_honest_not_perfect() -> None:
    micro = evaluate()["micro"]
    assert micro["precision"] >= 0.75
    assert micro["recall"] >= 0.70
    # The corpus includes known misses on purpose. A 1.0 score would mean we
    # deleted the hard cases instead of measuring them.
    assert micro["fn"] >= 1
    assert micro["f1"] < 1.0


def test_known_miss_ids_are_visible() -> None:
    ids = {item["id"] for item in evaluate()["failures"]}
    assert "miss_bare_name" in ids
    assert "miss_ssn_undashed" in ids
