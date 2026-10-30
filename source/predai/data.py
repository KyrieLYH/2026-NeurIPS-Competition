# -*- coding: utf-8 -*-
"""Load public parquet benches into competition-shaped records."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

import numpy as np
import pandas as pd

from predai.features import _s
from predai.paths import RAW_DATA

BENCH_ORDER = [
    "matharena",
    "mmdocrag",
    "multi_swebench",
    "real_webagents",
    "researchcodebench",
    "swe_rebench",
]

SUBJECT_KEYS = [
    "normalized_name",
    "provider",
    "release_date",
    "access_date",
    "harness",
    "harness_version",
    "reasoning_effort",
    "subject_features_extra",
]


def default_root() -> Path:
    """Original benchmark folders: datasets/raw/<bench>/*.parquet"""
    return RAW_DATA


def list_benches(root: Path | None = None) -> list[str]:
    from predai.tables import discover
    found = discover(root or default_root())
    return [b for b in BENCH_ORDER if b in found] + sorted(set(found) - set(BENCH_ORDER))


def _row_dict(row: pd.Series, columns: list[str]) -> dict[str, Any]:
    out = {}
    for col in columns:
        if col not in row.index:
            out[col] = ""
            continue
        val = row[col]
        if val is None or (isinstance(val, float) and pd.isna(val)):
            out[col] = ""
        else:
            out[col] = val
    return out


def binarize_label(response: Any, threshold: float = 0.5) -> float:
    """Legacy scalar helper: retain valid probabilities; reject missing/out-of-scale labels."""
    value = float(response)
    if not np.isfinite(value) or not 0 <= value <= 1:
        raise ValueError('Label must be finite in [0, 1]')
    return value


def load_bench_frames(bench: str, root: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from predai.tables import table_path
    root = root or default_root()
    return tuple(pd.read_parquet(table_path(root, bench, t))
                 for t in ('subjects', 'items', 'responses'))


def load_unified_frame(
    bench: str,
    root: Path | None = None,
    max_content_chars: int = 12000,
    label_threshold: float = 0.5,
    max_rows: int = 0,
) -> pd.DataFrame:
    from predai.data_v2 import load_snapshot
    df = load_snapshot(root or default_root(), label_policy='binary-benchmarks',
                       benches=[bench], max_content_chars=max_content_chars)
    df['response_raw'] = df['label']
    df['display_name'] = df['normalized_name']
    if max_rows and len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=0)
    return df.reset_index(drop=True)


def iter_records(
    bench: str,
    root: Path | None = None,
    max_content_chars: int = 12000,
    label_threshold: float = 0.5,
) -> Iterator[dict[str, Any]]:
    df = load_unified_frame(bench, root, max_content_chars, label_threshold)
    for rec in df.to_dict(orient="records"):
        subject, item = frame_to_subject_item(pd.Series(rec))
        yield {
            "benchmark_id": rec["benchmark_id"],
            "subject_id": str(rec["subject_id"]),
            "item_id": str(rec["item_id"]),
            "trial": int(rec["trial"]),
            "label": float(rec["label"]),
            "response_raw": float(rec["response_raw"]),
            "subject": subject,
            "item": item,
        }


def records_to_frame(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for r in records:
        s, i = r["subject"], r["item"]
        rows.append(
            {
                "benchmark_id": r["benchmark_id"],
                "subject_id": r["subject_id"],
                "item_id": r["item_id"],
                "trial": r["trial"],
                "label": r["label"],
                "response_raw": r["response_raw"],
                "normalized_name": s["normalized_name"],
                "provider": s["provider"],
                "release_date": s["release_date"],
                "access_date": s["access_date"],
                "harness": s["harness"],
                "harness_version": s["harness_version"],
                "reasoning_effort": s["reasoning_effort"],
                "subject_features_extra": s["subject_features_extra"],
                "display_name": s.get("display_name", ""),
                "item_content": i["item_content"],
                "item_features": i["item_features"],
                "interactors": i["interactors"],
            }
        )
    return pd.DataFrame(rows)


def frame_to_subject_item(row: pd.Series) -> tuple[dict, dict]:
    subject = {k: _s(row.get(k)) for k in SUBJECT_KEYS}
    item = {
        "item_content": _s(row.get("item_content")),
        "item_features": _s(row.get("item_features")),
        "interactors": _s(row.get("interactors")),
        "benchmark_id": _s(row.get("benchmark_id")),
    }
    return subject, item
