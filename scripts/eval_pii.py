"""
Precision / recall eval for the typed PII detector.

    python -m scripts.eval_pii
    python -m scripts.eval_pii --json
    python -m scripts.eval_pii --write-results

Matching: same entity type and character IoU >= 0.5 (first-come greedy).
The corpus is synthetic. Numbers are measured, not theoretical.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evals.corpus import iter_resolved  # noqa: E402

from scripts.pii_detect import ENTITY_TYPES, detect  # noqa: E402

IOU_THRESHOLD = 0.5
RESULTS_PATH = ROOT / "evals" / "results.json"


def _iou(a: dict[str, Any], b: dict[str, Any]) -> float:
    start = max(int(a["start"]), int(b["start"]))
    end = min(int(a["end"]), int(b["end"]))
    inter = max(0, end - start)
    if inter == 0:
        return 0.0
    union = int(a["end"]) - int(a["start"]) + int(b["end"]) - int(b["start"]) - inter
    return inter / union if union else 0.0


def match_entities(
    gold: list[dict[str, Any]],
    pred: list[dict[str, Any]],
    *,
    iou_threshold: float = IOU_THRESHOLD,
) -> tuple[int, int, int, list[tuple[dict[str, Any], dict[str, Any]]]]:
    """Greedy one-to-one match. Returns tp, fp, fn, pairs."""
    remaining = list(pred)
    pairs: list[tuple[dict[str, Any], dict[str, Any]]] = []
    tp = 0
    for g in gold:
        best_i = -1
        best_iou = 0.0
        for i, p in enumerate(remaining):
            if p["entity_type"] != g["entity_type"]:
                continue
            score = _iou(g, p)
            if score > best_iou:
                best_iou = score
                best_i = i
        if best_i >= 0 and best_iou >= iou_threshold:
            pairs.append((g, remaining.pop(best_i)))
            tp += 1
    fp = len(remaining)
    fn = len(gold) - tp
    return tp, fp, fn, pairs


def _prf(tp: int, fp: int, fn: int) -> dict[str, float]:
    prec = tp / (tp + fp) if (tp + fp) else 1.0
    rec = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
    return {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "support": tp + fn,
    }


def evaluate(documents: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    docs = documents if documents is not None else iter_resolved()
    per_type: dict[str, dict[str, int]] = {
        name: {"tp": 0, "fp": 0, "fn": 0} for name in ENTITY_TYPES
    }
    micro = {"tp": 0, "fp": 0, "fn": 0}
    per_doc: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for doc in docs:
        gold = list(doc["entities"])
        pred = [e.to_dict() for e in detect(doc["text"])]
        tp, fp, fn, pairs = match_entities(gold, pred)
        micro["tp"] += tp
        micro["fp"] += fp
        micro["fn"] += fn

        matched_gold = {id(g) for g, _ in pairs}
        matched_pred = {id(p) for _, p in pairs}
        gold_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        pred_by_type: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for g in gold:
            gold_by_type[g["entity_type"]].append(g)
        for p in pred:
            pred_by_type[p["entity_type"]].append(p)
        for name in ENTITY_TYPES:
            t_tp, t_fp, t_fn, _ = match_entities(gold_by_type[name], pred_by_type[name])
            per_type[name]["tp"] += t_tp
            per_type[name]["fp"] += t_fp
            per_type[name]["fn"] += t_fn

        misses = [g for g in gold if id(g) not in matched_gold]
        extras = [p for p in pred if id(p) not in matched_pred]
        if misses or extras:
            failures.append(
                {
                    "id": doc["id"],
                    "false_negatives": misses,
                    "false_positives": extras,
                    "notes": doc.get("notes") or "",
                }
            )
        per_doc.append(
            {
                "id": doc["id"],
                "gold": len(gold),
                "pred": len(pred),
                "tp": tp,
                "fp": fp,
                "fn": fn,
            }
        )

    by_type = {name: _prf(**counts) for name, counts in per_type.items()}
    measured = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    return {
        "date_utc": measured,
        "command": "python -m scripts.eval_pii",
        "iou_threshold": IOU_THRESHOLD,
        "documents": len(docs),
        "gold_entities": micro["tp"] + micro["fn"],
        "predicted_entities": micro["tp"] + micro["fp"],
        "micro": _prf(**micro),
        "per_entity_type": by_type,
        "failures": failures,
        "per_document": per_doc,
    }


def render_table(report: dict[str, Any]) -> str:
    lines = [
        f"PII detector eval  date={report['date_utc']}  docs={report['documents']}  "
        f"command=`{report['command']}`",
        "",
        f"{'type':<16} {'P':>7} {'R':>7} {'F1':>7} {'tp':>4} {'fp':>4} {'fn':>4} {'n':>4}",
        "-" * 64,
    ]
    for name, row in report["per_entity_type"].items():
        if row["support"] == 0 and row["fp"] == 0:
            continue
        lines.append(
            f"{name:<16} {row['precision']:7.3f} {row['recall']:7.3f} {row['f1']:7.3f} "
            f"{row['tp']:4d} {row['fp']:4d} {row['fn']:4d} {row['support']:4d}"
        )
    micro = report["micro"]
    lines.append("-" * 64)
    lines.append(
        f"{'MICRO':<16} {micro['precision']:7.3f} {micro['recall']:7.3f} {micro['f1']:7.3f} "
        f"{micro['tp']:4d} {micro['fp']:4d} {micro['fn']:4d} {micro['support']:4d}"
    )
    if report["failures"]:
        lines.append("")
        lines.append("Mismatches (known misses + false positives):")
        for item in report["failures"]:
            fn_types = ",".join(sorted({e['entity_type'] for e in item['false_negatives']})) or "-"
            fp_types = ",".join(sorted({e['entity_type'] for e in item['false_positives']})) or "-"
            note = f"  ({item['notes']})" if item["notes"] else ""
            lines.append(f"  - {item['id']}: FN[{fn_types}] FP[{fp_types}]{note}")
    return "\n".join(lines)


def write_results(report: dict[str, Any], path: Path = RESULTS_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = {
        "date_utc": report["date_utc"],
        "command": report["command"],
        "iou_threshold": report["iou_threshold"],
        "documents": report["documents"],
        "gold_entities": report["gold_entities"],
        "predicted_entities": report["predicted_entities"],
        "micro": report["micro"],
        "per_entity_type": report["per_entity_type"],
        "failure_ids": [f["id"] for f in report["failures"]],
    }
    path.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    return path


def write_corpus_jsonl(path: Path | None = None) -> Path:
    dest = path or (ROOT / "evals" / "pii_corpus.jsonl")
    dest.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(doc, ensure_ascii=False) for doc in iter_resolved()]
    dest.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Typed PII precision/recall eval")
    parser.add_argument("--json", action="store_true", help="Print the full JSON report")
    parser.add_argument("--write-results", action="store_true", help=f"Write {RESULTS_PATH}")
    parser.add_argument("--write-corpus", action="store_true", help="Refresh evals/pii_corpus.jsonl")
    args = parser.parse_args(argv)

    if args.write_corpus:
        write_corpus_jsonl()

    report = evaluate()
    if args.write_results:
        write_results(report)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print(render_table(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
